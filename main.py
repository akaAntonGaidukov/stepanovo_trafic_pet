from collections import defaultdict, deque
from typing import Tuple, List, Optional
import asyncio
import websockets
import av
import io
import logging
from ultralytics import YOLO
import cv2
import numpy as np
from contextlib import suppress
import time
from datetime import datetime
import os
import asyncpg  
# Constants
BUFFER_SIZE = 60  # Number of frames to accumulate before reporting
CLASS_NAMES = {0: "SB", 1: "SKI", 2: "LIFT"}
MASK_PATH = "mask.png"  # Replace with your mask path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.db_host = os.getenv('DB_HOST', 'db')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_user = os.getenv('DB_USER', 'postgres')
        self.db_password = os.getenv('DB_PASSWORD', 'postgres')
        self.db_name = os.getenv('DB_NAME', 'detection_db')
        
    async def connect(self):
        self.conn = await asyncpg.connect(
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_password,
            database=self.db_name,
            ssl=False
        )
        await self.create_table()

    async def create_table(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS detection_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            weekday INTEGER NOT NULL,
            lifts INTEGER NOT NULL,
            skis INTEGER NOT NULL,
            sbs INTEGER NOT NULL,
            processing_time REAL NOT NULL,
            fps REAL NOT NULL
        );
        """
        await self.conn.execute(create_table_query)

    async def insert_record(self, lifts, skis, sbs, processing_time, fps):
        now = datetime.now()
        insert_query = """
        INSERT INTO detection_logs (timestamp, weekday, lifts, skis, sbs, processing_time, fps)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        await self.conn.execute(
            insert_query,
            now,
            now.weekday(),
            lifts,
            skis,
            sbs,
            processing_time,
            fps
        )

    async def close(self):
        if self.conn:
            await self.conn.close()

class DetectionProcessor:
    def __init__(self):
        self.model = YOLO("yolo_s_best_result_3000.pt")
        self.frame_buffer = deque(maxlen=BUFFER_SIZE)
        self.count_history = deque(maxlen=1000)
        
        # Track lift centers instead of full boxes
        self.last_lift_center: Optional[np.ndarray] = None
        self.movement_threshold = 700  # Adjust based on your needs
        self.current_lift_id = 0
        self.seen_lifts = set()

    def _get_center(self, box: np.ndarray) -> np.ndarray:
        """Get center coordinates of a bounding box as numpy array"""
        if box.shape != (4,):
            raise ValueError(f"Invalid box shape: {box.shape}")
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    def _boxes_intersect(self, box1: np.ndarray, box2: np.ndarray) -> bool:
        """Check if two bounding boxes intersect"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

    def _count_associated_objects(self, lift_box: np.ndarray, 
                                ski_boxes: List[np.ndarray], 
                                sb_boxes: List[np.ndarray]) -> Tuple[int, int]:
        """Count objects touching the lift, max 2 total (closest)"""
        associated = []

        # Find intersecting skis and SBs
        for box in ski_boxes:
            if self._boxes_intersect(lift_box, box):
                associated.append(('ski', self._get_center(box)))

        for box in sb_boxes:
            if self._boxes_intersect(lift_box, box):
                associated.append(('sb', self._get_center(box)))

        # Get lift center
        lift_center = self._get_center(lift_box)

        # Sort all associated objects by distance and keep closest 2
        sorted_combined = sorted(
            associated,
            key=lambda x: np.linalg.norm(x[1] - lift_center)
        )[:2]

        # Count types among closest two
        ski_count = sum(1 for obj_type, _ in sorted_combined if obj_type == 'ski')
        sb_count = sum(1 for obj_type, _ in sorted_combined if obj_type == 'sb')

        return ski_count, sb_count

    def _track_lift(self, centers: List[np.ndarray]) -> Optional[np.ndarray]:
        """Track lift movement with hysteresis"""
        if not centers:
            self.last_lift_center = None
            return None
        
        # Find closest center to last known position
        if self.last_lift_center is not None:
            distances = [np.linalg.norm(c - self.last_lift_center) for c in centers]
            closest_idx = np.argmin(distances)
            
            if distances[closest_idx] < self.movement_threshold:
                # Update position for same lift
                self.last_lift_center = centers[closest_idx]
                return self.last_lift_center
        
        # New lift detection
        if self.last_lift_center is None:
            # Find closest to frame center
            frame_center = np.array([640, 360])  # Adjust to your frame size
            distances = [np.linalg.norm(c - frame_center) for c in centers]
            closest_idx = np.argmin(distances)
            
        self.last_lift_center = centers[closest_idx]
        self.current_lift_id += 1
        return self.last_lift_center

    def _filter_lifts(self, lift_boxes: List[np.ndarray], frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Get tracked lift center or None"""
        if not lift_boxes:
            return None
            
        centers = [self._get_center(box) for box in lift_boxes]
        tracked_center = self._track_lift(centers)
        
        # Only count new lifts that stay in view for multiple frames
        if tracked_center is not None and self.current_lift_id not in self.seen_lifts:
            if len(self.frame_buffer) > 5:  # Require 5-frame confirmation
                self.seen_lifts.add(self.current_lift_id)
                return tracked_center
        return None
    
    async def process_frame(self, frame: np.ndarray) -> None:
        """Process a single frame and update counts."""
        try:
            # Preprocess frame
            height, width = frame.shape[:2]
            right_half = frame[:, width // 2 :]
            mask_resized = cv2.resize(
                cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE),
                (right_half.shape[1], right_half.shape[0]),
            )
            processed_frame = cv2.bitwise_and(right_half, right_half, mask=mask_resized)

            # Run detection
            results = self.model.predict(
                processed_frame, iou=0.4, conf=0.1, imgsz=640, verbose=False
            )[0]

            # Parse results
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            # Validate bounding boxes
            valid_boxes = []
            valid_classes = []
            for box, cls in zip(boxes, classes):
                if len(box) == 4:  # Ensure the box has 4 values (x1, y1, x2, y2)
                    valid_boxes.append(box)
                    valid_classes.append(cls)

            # Separate detections by class
            lifts = [box for box, cls in zip(valid_boxes, valid_classes) if cls == 2]
            skis = [box for box, cls in zip(valid_boxes, valid_classes) if cls == 1]
            sbs = [box for box, cls in zip(valid_boxes, valid_classes) if cls == 0]

            # Process lifts
            filtered_center = self._filter_lifts(lifts, processed_frame.shape)
            
            counts = defaultdict(int)
            if filtered_center is not None:
                # Find closest box to tracked center
                closest_box = min(lifts, key=lambda b: np.linalg.norm(self._get_center(b) - filtered_center))
                ski_count, sb_count = self._count_associated_objects(closest_box, skis, sbs)
                counts["lifts"] = 1
                counts["skis"] = ski_count
                counts["sbs"] = sb_count

            self.frame_buffer.append(counts)

        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            
class VideoProcessor:
    def __init__(self, ws_url: str, headers: dict):
        self.ws_url = ws_url
        self.headers = headers
        self.frame_queue = asyncio.Queue(maxsize=30)  # Buffer for 1 second at 30 FPS
        self._restart_counts = deque(maxlen=5)
        self._shutdown = False

        # Initialize detection processor
        self.detection_processor = DetectionProcessor()
        self.frame_counter = 0
        self.batch_start_time: Optional[float] = None
        
        # Initialize database processor
        self.db_manager = DatabaseManager()

    async def start(self):
            """Main entry point to start processing loop."""
            await self.db_manager.connect()
            try:
                while not self._shutdown:
                    await self._run_with_restart_backoff()
            except Exception as e:
                logging.error(f"Unexpected error in main loop: {e}")
            finally:
                await self.db_manager.close()

    async def _run_with_restart_backoff(self):
        """Handle restart logic with progressive delays."""
        attempts = len(self._restart_counts)
        wait_times = [0, 10, 60, 300, 1800]

        if attempts > 0:
            wait = wait_times[min(attempts), len(wait_times) - 1]
            logging.info(f"Restarting in {wait} seconds (attempt {attempts + 1})")
            await asyncio.sleep(wait)

        try:
            async with websockets.connect(
                self.ws_url,
                extra_headers=self.headers,
                ping_timeout=30,
                close_timeout=10,
            ) as ws:
                self._restart_counts.clear()
                await self._process_stream(ws)
        except Exception as e:
            logging.error(f"Stream connection failed: {e}")
            self._restart_counts.append(time.time())
            raise

    async def _process_stream(self, ws):
        """Handle incoming stream messages."""
        moov_data = None
        async for message in ws:
            try:
                if moov_data is None:
                    moov_data = await self._handle_init_segment(message)
                    continue

                if self.frame_queue.full():
                    # Drop oldest frame if queue is full
                    with suppress(asyncio.QueueEmpty):
                        self.frame_queue.get_nowait()

                await self.frame_queue.put((moov_data, message))
            except Exception as e:
                logging.error(f"Error processing message: {e}")

    async def _handle_init_segment(self, message) -> Optional[bytearray]:
        """Handle initial MP4 moov atom."""
        if isinstance(message, str) or len(message) < 8:
            return None

        moov_data = bytearray(message)
        if b"moov" not in moov_data:
            return None

        logging.info("Received initialization segment")
        return moov_data

    async def frame_generator(self):
        """Async generator yielding decoded video frames."""
        while not self._shutdown:
            try:
                moov_data, fragment = await self.frame_queue.get()
                container = av.open(
                    io.BytesIO(moov_data + fragment), format="mp4", timeout=5
                )

                for frame in container.decode(video=0):
                    yield frame.to_ndarray(format="bgr24")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Frame decoding failed: {e}")

    async def process_frames(self):
        """Process frames and yield results to DetectionProcessor."""
        async for frame in self.frame_generator():
            try:
                # Start timing for the first frame in batch
                if self.frame_counter % BUFFER_SIZE == 0:
                    self.batch_start_time = time.time()

                # Process frame through detection processor
                await self.detection_processor.process_frame(frame)
                self.frame_counter += 1

                # Print results every BUFFER_SIZE frames
                if self.frame_counter % BUFFER_SIZE == 0:
                    elapsed_time = time.time() - self.batch_start_time
                    await self._print_batch_results(elapsed_time)

            except Exception as e:
                logging.error(f"Frame processing error: {e}")

    async def _print_batch_results(self, elapsed_time: float):
            """Print and store results for the last batch of frames."""
            if not self.detection_processor.frame_buffer:
                return

            # Aggregate results
            total = defaultdict(int)
            for counts in self.detection_processor.frame_buffer:
                for k, v in counts.items():
                    total[k] += v

            # Calculate FPS
            fps = BUFFER_SIZE / elapsed_time if elapsed_time > 0 else 0

            # Print results
            logging.info(
                f"Batch Results (last {BUFFER_SIZE} frames, {elapsed_time:.2f}s, {fps:.1f} FPS):\n"
                f"  Lifts: {total.get('lifts', 0)}\n"
                f"  Skis: {total.get('skis', 0)}\n"
                f"  SBs: {total.get('sbs', 0)}\n"
                f"  Processing time: {elapsed_time:.2f}s\n"
                "----------------------------------------"
            )

            # Store in database
            try:
                await self.db_manager.insert_record(
                    lifts=total.get('lifts', 0),
                    skis=total.get('skis', 0),
                    sbs=total.get('sbs', 0),
                    processing_time=elapsed_time,
                    fps=fps
                )
            except Exception as e:
                logging.error(f"Database insertion error: {e}")

            # Clear buffer for next batch
            self.detection_processor.frame_buffer.clear()


    async def shutdown(self):
        """Graceful shutdown."""
        self._shutdown = True
        with suppress(asyncio.QueueEmpty):
            while True:
                self.frame_queue.get_nowait()


async def main():
    # Example configuration
    devcode = "fcbe5ab2393d44ffa8105867950d366a"
    ws_url = f"wss://sr-171-25-233-93.ipeye.ru/ws/mp4/live?name={devcode}&mode=live"
    config = {
        "ws_url": ws_url,
        "headers": {
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Sec-WebSocket-Version": "13",
            "Origin": "https://ipeye.ru",
        },
    }

    processor = VideoProcessor(**config)

    try:
        await asyncio.gather(
            processor.start(),
            processor.process_frames(),  # Use the new processing method
        )
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        await processor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())