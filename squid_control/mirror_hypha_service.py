import argparse
import asyncio
import fractions
import json
import logging
import logging.handlers
import os
import time
import traceback

# WebRTC imports
import aiohttp

# Image processing imports
import cv2
import dotenv
import numpy as np
from aiortc import MediaStreamTrack
from av import VideoFrame
from hypha_rpc import connect_to_server, login, register_rtc_service

dotenv.load_dotenv()  
ENV_FILE = dotenv.find_dotenv()  
if ENV_FILE:
    dotenv.load_dotenv(ENV_FILE)  

# Set up logging
def setup_logging(log_file="mirror_squid_control_service.log", max_bytes=100000, backup_count=3):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()


class MicroscopeVideoTrack(MediaStreamTrack):
    """
    A video stream track that provides real-time microscope images.
    """

    kind = "video"

    def __init__(self, local_service, parent_service=None):
        super().__init__()
        if local_service is None:
            raise ValueError("local_service cannot be None when initializing MicroscopeVideoTrack")
        self.local_service = local_service
        self.parent_service = parent_service  # Reference to MirrorMicroscopeService for data channel access
        self.count = 0
        self.running = True
        self.start_time = None
        self.fps = 5 # Target FPS for WebRTC stream
        self.frame_width = 750
        self.frame_height = 750
        logger.info("MicroscopeVideoTrack initialized with local_service")

    def draw_crosshair(self, img, center_x, center_y, size=20, color=[255, 255, 255]):
        """Draw a crosshair at the specified position"""
        height, width = img.shape[:2]

        # Horizontal line
        if 0 <= center_y < height:
            start_x = max(0, center_x - size)
            end_x = min(width, center_x + size)
            img[center_y, start_x:end_x] = color

        # Vertical line
        if 0 <= center_x < width:
            start_y = max(0, center_y - size)
            end_y = min(height, center_y + size)
            img[start_y:end_y, center_x] = color

    async def recv(self):
        if not self.running:
            logger.warning("MicroscopeVideoTrack: recv() called but track is not running")
            raise Exception("Track stopped")

        try:
            if self.start_time is None:
                self.start_time = time.time()

            # Time the entire frame processing (including sleep)
            frame_start_time = time.time()

            # Calculate and perform FPS throttling sleep
            next_frame_time = self.start_time + (self.count / self.fps)
            sleep_duration = next_frame_time - time.time()
            sleep_start = time.time()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            sleep_end = time.time()
            actual_sleep_time = (sleep_end - sleep_start) * 1000  # Convert to ms

            # Start timing actual processing after sleep
            processing_start_time = time.time()

            # Check if local_service is still available
            if self.local_service is None:
                logger.error("MicroscopeVideoTrack: local_service is None")
                raise Exception("Local service not available")

            # Time getting the video frame from local service
            get_frame_start = time.time()
            frame_data = await self.local_service.get_video_frame(
                frame_width=self.frame_width,
                frame_height=self.frame_height
            )
            get_frame_end = time.time()
            get_frame_latency = (get_frame_end - get_frame_start) * 1000  # Convert to ms

            # Extract stage position from frame metadata
            stage_position = None
            if isinstance(frame_data, dict) and 'metadata' in frame_data:
                stage_position = frame_data['metadata'].get('stage_position')
                logger.debug(f"Frame {self.count}: Found stage_position in metadata: {stage_position}")
            else:
                logger.debug(f"Frame {self.count}: No metadata found in frame_data, keys: {list(frame_data.keys()) if isinstance(frame_data, dict) else 'not dict'}")

            # Handle new JPEG format returned by get_video_frame
            if isinstance(frame_data, dict) and 'data' in frame_data:
                # New format: dictionary with JPEG data
                jpeg_data = frame_data['data']
                frame_size_bytes = frame_data.get('size_bytes', len(jpeg_data))
                #compression_ratio = frame_data.get('compression_ratio', 1.0)

                # Decode JPEG data to numpy array
                decode_start = time.time()
                if isinstance(jpeg_data, bytes):
                    # Convert bytes to numpy array for cv2.imdecode
                    jpeg_np = np.frombuffer(jpeg_data, dtype=np.uint8)
                    # Decode JPEG to BGR format (OpenCV default)
                    processed_frame_bgr = cv2.imdecode(jpeg_np, cv2.IMREAD_COLOR)
                    if processed_frame_bgr is None:
                        raise Exception("Failed to decode JPEG data")
                    # Convert BGR to RGB for VideoFrame
                    processed_frame = cv2.cvtColor(processed_frame_bgr, cv2.COLOR_BGR2RGB)
                else:
                    raise Exception(f"Unexpected JPEG data type: {type(jpeg_data)}")
                decode_end = time.time()
                decode_latency = (decode_end - decode_start) * 1000  # Convert to ms
                print(f"Frame {self.count} decode time: {decode_latency:.2f}ms")
            else:
                # Fallback for old format (numpy array)
                processed_frame = frame_data
                if hasattr(processed_frame, 'nbytes'):
                    frame_size_bytes = processed_frame.nbytes
                else:
                    import sys  # noqa: PLC0415
                    frame_size_bytes = sys.getsizeof(processed_frame)

                frame_size_kb = frame_size_bytes / 1024
                print(f"Frame {self.count} raw data size: {frame_size_kb:.2f} KB ({frame_size_bytes} bytes)")

            # Time processing the frame
            process_start = time.time()
            current_time = time.time()
            # Use a 90kHz timebase, common for video, to provide accurate frame timing.
            # This prevents video from speeding up if frame acquisition is slow.
            time_base = fractions.Fraction(1, 90000)
            pts = int((current_time - self.start_time) * time_base.denominator)

            new_video_frame = VideoFrame.from_ndarray(processed_frame, format="rgb24")
            new_video_frame.pts = pts
            new_video_frame.time_base = time_base
            process_end = time.time()
            process_latency = (process_end - process_start) * 1000  # Convert to ms

            # SEND METADATA VIA WEBRTC DATA CHANNEL
            # Send metadata through data channel instead of embedding in video frame
            if stage_position and self.parent_service:
                try:
                    # Create frame metadata including stage position
                    frame_metadata = {
                        'stage_position': stage_position,
                        'timestamp': current_time,
                        'frame_count': self.count
                    }
                    # Add any additional metadata from frame_data if available
                    if isinstance(frame_data, dict) and 'metadata' in frame_data:
                        frame_metadata.update(frame_data['metadata'])

                    metadata_json = json.dumps(frame_metadata)
                    # Send metadata via WebRTC data channel
                    asyncio.create_task(self._send_metadata_via_datachannel(metadata_json))
                    logger.debug(f"Sent metadata via data channel: {len(metadata_json)} bytes")
                except Exception as e:
                    logger.warning(f"Failed to send metadata via data channel: {e}")

            # Calculate processing and total latencies
            processing_end_time = time.time()
            processing_latency = (processing_end_time - processing_start_time) * 1000  # Convert to ms
            total_frame_latency = (processing_end_time - frame_start_time) * 1000  # Convert to ms

            # Print timing information every frame (you can adjust frequency as needed)
            if isinstance(frame_data, dict) and 'data' in frame_data:
                print(f"Frame {self.count} timing: sleep={actual_sleep_time:.2f}ms, get_video_frame={get_frame_latency:.2f}ms, decode={decode_latency:.2f}ms, process={process_latency:.2f}ms, processing_total={processing_latency:.2f}ms, total_with_sleep={total_frame_latency:.2f}ms")
            else:
                print(f"Frame {self.count} timing: sleep={actual_sleep_time:.2f}ms, get_video_frame={get_frame_latency:.2f}ms, process={process_latency:.2f}ms, processing_total={processing_latency:.2f}ms, total_with_sleep={total_frame_latency:.2f}ms")

            if self.count % (self.fps * 5) == 0:  # Log every 5 seconds
                duration = current_time - self.start_time
                if duration > 0:
                    actual_fps = (self.count + 1) / duration
                    logger.info(f"MicroscopeVideoTrack: Sent frame {self.count}, actual FPS: {actual_fps:.2f}")
                else:
                    logger.info(f"MicroscopeVideoTrack: Sent frame {self.count}")

            self.count += 1
            return new_video_frame

        except Exception as e:
            logger.error(f"MicroscopeVideoTrack: Error in recv(): {e}", exc_info=True)
            self.running = False
            raise

    def stop(self):
        logger.info("MicroscopeVideoTrack stop() called.")
        self.running = False

    async def _send_metadata_via_datachannel(self, metadata_json):
        """Send metadata via WebRTC data channel"""
        try:
            if (self.parent_service and 
                hasattr(self.parent_service, 'metadata_data_channel') and 
                self.parent_service.metadata_data_channel):
                if self.parent_service.metadata_data_channel.readyState == 'open':
                    self.parent_service.metadata_data_channel.send(metadata_json)
                    logger.debug(f"Metadata sent via data channel: {len(metadata_json)} bytes")
                else:
                    logger.debug(f"Data channel not ready, state: {self.parent_service.metadata_data_channel.readyState}")
        except Exception as e:
            logger.warning(f"Error sending metadata via data channel: {e}")

class MirrorMicroscopeService:
    def __init__(self):
        self.login_required = True
        # Connection to cloud service
        self.cloud_server_url = "https://hypha.aicell.io"
        self.cloud_workspace = "reef-imaging"
        self.cloud_token = os.environ.get("REEF_WORKSPACE_TOKEN")
        self.cloud_service_id = "mirror-microscope-control-squid-1"
        self.cloud_server = None
        self.cloud_service = None  # Add reference to registered cloud service

        # Connection to local service
        self.local_server_url = "http://reef.dyn.scilifelab.se:9527"
        self.local_token = os.environ.get("REEF_LOCAL_TOKEN")
        self.local_service_id = "microscope-control-squid-1"
        self.local_server = None
        self.local_service = None
        self.video_track = None

        # Video streaming state
        self.is_streaming = False
        self.webrtc_service_id = None
        self.webrtc_connected = False
        self.metadata_data_channel = None

        # Setup task tracking
        self.setup_task = None

        # Store dynamically created mirror methods
        self.mirrored_methods = {}

    async def connect_to_local_service(self):
        """Connect to the local microscope service"""
        try:
            logger.info(f"Connecting to local service at {self.local_server_url}")
            self.local_server = await connect_to_server({
                "server_url": self.local_server_url, 
                "token": self.local_token,
                "ping_interval": None
            })

            # Connect to the local service
            self.local_service = await self.local_server.get_service(self.local_service_id)
            logger.info(f"Successfully connected to local service {self.local_service_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to local service: {e}")
            self.local_service = None
            self.local_server = None
            return False

    async def cleanup_cloud_service(self):
        """Clean up the cloud service registration"""
        try:
            if self.cloud_service:
                logger.info(f"Unregistering cloud service {self.cloud_service_id}")
                # Try to unregister the service
                try:
                    await self.cloud_server.unregister_service(self.cloud_service_id)
                    logger.info(f"Successfully unregistered cloud service {self.cloud_service_id}")
                except Exception as e:
                    logger.warning(f"Failed to unregister cloud service {self.cloud_service_id}: {e}")

                self.cloud_service = None

            # Clear mirrored methods
            self.mirrored_methods.clear()
            logger.info("Cleared mirrored methods")

        except Exception as e:
            logger.error(f"Error during cloud service cleanup: {e}")

    def _create_mirror_method(self, method_name, local_method):
        """Create a mirror method that forwards calls to the local service"""
        async def mirror_method(*args, **kwargs):
            try:
                if self.local_service is None:
                    logger.warning(f"Local service is None when calling {method_name}, attempting to reconnect")
                    success = await self.connect_to_local_service()
                    if not success or self.local_service is None:
                        raise Exception("Failed to connect to local service")

                # Forward the call to the local service
                result = await local_method(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Failed to call {method_name}: {e}")
                raise e

        return mirror_method

    def _get_mirrored_methods(self):
        """Dynamically create mirror methods for all callable methods in local_service"""
        if self.local_service is None:
            logger.warning("Cannot create mirror methods: local_service is None")
            return {}

        logger.info(f"Creating mirror methods for local service {self.local_service_id}")
        logger.info(f"Local service type: {type(self.local_service)}")
        logger.info(f"Local service attributes: {list(dir(self.local_service))}")

        mirrored_methods = {}
        
        # Methods to exclude from mirroring (these are handled specially)
        excluded_methods = {
            'name', 'id', 'config', 'type',  # Service metadata
            '__class__', '__doc__', '__dict__', '__module__',  # Python internals
        }
        
        # Get all attributes from the local service
        for attr_name in dir(self.local_service):
            if attr_name.startswith('_') or attr_name in excluded_methods:
                logger.debug(f"Skipping attribute: {attr_name} (excluded or private)")
                continue
                
            attr = getattr(self.local_service, attr_name)
            
            # Check if it's callable (a method)
            if callable(attr):
                logger.info(f"Creating mirror method for: {attr_name}")
                mirrored_methods[attr_name] = self._create_mirror_method(attr_name, attr)
            else:
                logger.debug(f"Skipping non-callable attribute: {attr_name}")
        
        logger.info(f"Total mirrored methods created: {len(mirrored_methods)}")
        logger.info(f"Mirrored method names: {list(mirrored_methods.keys())}")
        return mirrored_methods

    async def check_service_health(self):
        """Check if the service is healthy and rerun setup if needed"""
        logger.info("Starting service health check task")
        while True:
            try:
                # Try to get the service status
                if self.cloud_service_id and self.cloud_server:
                    try:
                        service = await self.cloud_server.get_service(self.cloud_service_id)
                        # Try a simple operation to verify service is working
                        ping_result = await asyncio.wait_for(service.ping(), timeout=10)
                        if ping_result != "pong":
                            logger.error(f"Cloud service health check failed: {ping_result}")
                            raise Exception("Cloud service not healthy")
                    except Exception as e:
                        logger.error(f"Cloud service health check failed: {e}")
                        raise Exception(f"Cloud service not healthy: {e}")
                else:
                    logger.info("Cloud service ID or server not set, waiting for service registration")
                    
                # Always check local service regardless of whether it's None
                try:
                    if self.local_service is None:
                        logger.info("Local service connection lost, attempting to reconnect")
                        success = await self.connect_to_local_service()
                        if not success or self.local_service is None:
                            raise Exception("Failed to connect to local service")
                    
                    #logger.info("Checking local service health...")
                    local_ping_result = await asyncio.wait_for(self.local_service.ping(), timeout=10)
                    #logger.info(f"Local service response: {local_ping_result}")
                    
                    if local_ping_result != "pong":
                        logger.error(f"Local service health check failed: {local_ping_result}")
                        raise Exception("Local service not healthy")
                    
                    #logger.info("Local service health check passed")
                except Exception as e:
                    logger.error(f"Local service health check failed: {e}")
                    self.local_service = None  # Reset connection so it will reconnect next time
                    raise Exception(f"Local service not healthy: {e}")
            except Exception as e:
                logger.error(f"Service health check failed: {e}")
                logger.info("Attempting to clean up and rerun setup...")
                
                # Clean up everything properly
                try:
                    # First, clean up the cloud service
                    await self.cleanup_cloud_service()
                    
                    # Then disconnect from servers
                    if self.cloud_server:
                        await self.cloud_server.disconnect()
                    if self.local_server:
                        await self.local_server.disconnect()
                    if self.setup_task:
                        self.setup_task.cancel()  # Cancel the previous setup task
                except Exception as disconnect_error:
                    logger.error(f"Error during cleanup: {disconnect_error}")
                finally:
                    self.cloud_server = None
                    self.cloud_service = None
                    self.local_server = None
                    self.local_service = None
                    self.mirrored_methods.clear()

                # Retry setup with exponential backoff
                retry_count = 0
                max_retries = 50
                base_delay = 10
                
                while retry_count < max_retries:
                    try:
                        delay = base_delay * (2 ** min(retry_count, 5))  # Cap at 32 * base_delay
                        logger.info(f"Retrying setup in {delay} seconds (attempt {retry_count + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        
                        # Rerun the setup method
                        self.setup_task = asyncio.create_task(self.setup())
                        await self.setup_task
                        logger.info("Setup successful after reconnection")
                        break  # Exit the loop if setup is successful
                    except Exception as setup_error:
                        retry_count += 1
                        logger.error(f"Failed to rerun setup (attempt {retry_count}/{max_retries}): {setup_error}")
                        if retry_count >= max_retries:
                            logger.error("Max retries reached, giving up on setup")
                            await asyncio.sleep(60)  # Wait longer before next health check cycle
                            break
            
            await asyncio.sleep(10)  # Check more frequently (was 30)

    async def start_hypha_service(self, server):
        """Start the Hypha service with dynamically mirrored methods"""
        self.cloud_server = server
        
        # Ensure we have a connection to the local service
        if self.local_service is None:
            logger.info("Local service not connected, attempting to connect before creating mirror methods")
            success = await self.connect_to_local_service()
            if not success:
                raise Exception("Cannot start Hypha service without local service connection")
        
        # Get the mirrored methods from the current local service
        self.mirrored_methods = self._get_mirrored_methods()
        
        # Base service configuration with core methods
        service_config = {
            "name": "Mirror Microscope Control Service",
            "id": self.cloud_service_id,
            "config": {
                "visibility": "public",
                "run_in_executor": True
            },
            "type": "echo",
            "ping": self.ping,
        }
        
        # Add all mirrored methods to the service configuration
        service_config.update(self.mirrored_methods)
        
        # Register the service
        self.cloud_service = await server.register_service(service_config)

        logger.info(
            f"Mirror service (service_id={self.cloud_service_id}) started successfully with {len(self.mirrored_methods)} mirrored methods, available at {self.cloud_server_url}/services"
        )

        logger.info(f'You can use this service using the service id: {self.cloud_service.id}')
        id = self.cloud_service.id.split(":")[1]

        logger.info(f"You can also test the service via the HTTP proxy: {self.cloud_server_url}/{server.config.workspace}/services/{id}")

    async def start_webrtc_service(self, server, webrtc_service_id_arg):
        self.webrtc_service_id = webrtc_service_id_arg 
        
        async def on_init(peer_connection):
            logger.info("WebRTC peer connection initialized")
            # Mark as connected when peer connection starts
            self.webrtc_connected = True
            
            # Create data channel for metadata transmission
            self.metadata_data_channel = peer_connection.createDataChannel("metadata", ordered=True)
            logger.info("Created metadata data channel")
            
            @self.metadata_data_channel.on("open")
            def on_data_channel_open():
                logger.info("Metadata data channel opened")
            
            @self.metadata_data_channel.on("close")
            def on_data_channel_close():
                logger.info("Metadata data channel closed")
            
            @self.metadata_data_channel.on("error")
            def on_data_channel_error(error):
                logger.error(f"Metadata data channel error: {error}")
            
            @peer_connection.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"WebRTC connection state changed to: {peer_connection.connectionState}")
                if peer_connection.connectionState in ["closed", "failed", "disconnected"]:
                    # Mark as disconnected
                    self.webrtc_connected = False
                    self.metadata_data_channel = None
                    self.local_service.off_illumination()
                    logger.info("Illumination closed")
                    if self.video_track and self.video_track.running:
                        logger.info(f"Connection state is {peer_connection.connectionState}. Stopping video track.")
                        self.video_track.stop()
                elif peer_connection.connectionState in ["connected"]:
                    # Mark as connected
                    self.webrtc_connected = True
            
            @peer_connection.on("track")
            def on_track(track):
                logger.info(f"Track {track.kind} received from client")
                
                if self.video_track and self.video_track.running:
                    self.video_track.stop() 
                
                # Ensure local_service is available before creating video track
                if self.local_service is None:
                    logger.error("Cannot create video track: local_service is not available")
                    return
                
                try:
                    self.local_service.on_illumination()
                    logger.info("Illumination opened")
                    self.video_track = MicroscopeVideoTrack(self.local_service, self)
                    peer_connection.addTrack(self.video_track)
                    logger.info("Added MicroscopeVideoTrack to peer connection")
                except Exception as e:
                    logger.error(f"Failed to create video track: {e}")
                    return
                
                @track.on("ended")
                def on_ended():
                    logger.info(f"Client track {track.kind} ended")
                    self.local_service.off_illumination()
                    logger.info("Illumination closed")
                    if self.video_track:
                        logger.info("Stopping MicroscopeVideoTrack.")
                        self.video_track.stop()  # Now synchronous
                        self.video_track = None
                    self.metadata_data_channel = None

        ice_servers = await self.fetch_ice_servers()
        if not ice_servers:
            logger.warning("Using fallback ICE servers")
            ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

        try:
            await register_rtc_service(
                server,
                service_id=self.webrtc_service_id,
                config={
                    "visibility": "public",
                    "ice_servers": ice_servers,
                    "on_init": on_init,
                },
            )
            logger.info(f"WebRTC service registered with id: {self.webrtc_service_id}")
        except Exception as e:
            logger.error(f"Failed to register WebRTC service ({self.webrtc_service_id}): {e}")
            if "Service already exists" in str(e):
                logger.info(f"WebRTC service {self.webrtc_service_id} already exists. Attempting to retrieve it.")
                try:
                    _ = await server.get_service(self.webrtc_service_id)
                    logger.info(f"Successfully retrieved existing WebRTC service: {self.webrtc_service_id}")
                except Exception as get_e:
                    logger.error(f"Failed to retrieve existing WebRTC service {self.webrtc_service_id}: {get_e}")
                    raise
            else:
                raise

    async def setup(self):
        # Connect to cloud workspace
        logger.info(f"Connecting to cloud workspace {self.cloud_workspace} at {self.cloud_server_url}")
        server = await connect_to_server({
            "server_url": self.cloud_server_url, 
            "token": self.cloud_token, 
            "workspace": self.cloud_workspace,
            "ping_interval": None
        })
        
        # Connect to local service first (needed to get available methods)
        logger.info("Connecting to local service before setting up mirror service")
        success = await self.connect_to_local_service()
        if not success or self.local_service is None:
            raise Exception("Failed to connect to local service during setup")
        
        # Verify local service is working
        try:
            ping_result = await asyncio.wait_for(self.local_service.ping(), timeout=10)
            if ping_result != "pong":
                raise Exception(f"Local service verification failed: {ping_result}")
            logger.info("Local service connection verified successfully")
        except Exception as e:
            logger.error(f"Local service verification failed: {e}")
            raise Exception(f"Local service not responding properly: {e}")
        
        # Small delay to ensure local service is fully ready
        await asyncio.sleep(1)
        
        # Start the cloud service with mirrored methods
        logger.info("Starting cloud service with mirrored methods")
        await self.start_hypha_service(server)
        
        # Start the WebRTC service
        self.webrtc_service_id = f"video-track-{self.local_service_id}"
        logger.info(f"Starting WebRTC service with id: {self.webrtc_service_id}")
        await self.start_webrtc_service(server, self.webrtc_service_id)
        
        logger.info("Setup completed successfully")

    def ping(self):
        """Ping function for health checks"""
        return "pong"

    async def fetch_ice_servers(self):
        """Fetch ICE servers from the coturn service"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://ai.imjoy.io/public/services/coturn/get_rtc_ice_servers') as response:
                    if response.status == 200:
                        ice_servers = await response.json()
                        logger.info("Successfully fetched ICE servers")
                        return ice_servers
                    else:
                        logger.warning(f"Failed to fetch ICE servers, status: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching ICE servers: {e}")
            return None

    def start_video_streaming(self, context=None):
        """Start WebRTC video streaming"""
        try:
            if not self.is_streaming:
                self.is_streaming = True
                logger.info("Video streaming started")
                return {"status": "streaming_started", "message": "WebRTC video streaming has been started"}
            else:
                return {"status": "already_streaming", "message": "Video streaming is already active"}
        except Exception as e:
            logger.error(f"Failed to start video streaming: {e}")
            raise e

    def stop_video_streaming(self, context=None):
        """Stop WebRTC video streaming"""
        try:
            if self.is_streaming:
                self.is_streaming = False
                if self.video_track:
                    self.video_track.running = False
                logger.info("Video streaming stopped")
                return {"status": "streaming_stopped", "message": "WebRTC video streaming has been stopped"}
            else:
                return {"status": "not_streaming", "message": "Video streaming is not currently active"}
        except Exception as e:
            logger.error(f"Failed to stop video streaming: {e}")
            raise e


    


    async def set_video_fps(self, fps=5, context=None):
        """Special method to set video FPS for both WebRTC and local service"""
        try:
            if self.local_service is None:
                await self.connect_to_local_service()
            
            # Update WebRTC video track FPS if active
            if self.video_track and self.video_track.running:
                old_webrtc_fps = self.video_track.fps
                self.video_track.fps = fps
                logger.info(f"WebRTC video track FPS updated from {old_webrtc_fps} to {fps}")
            
            # Forward call to local service if it has this method
            if hasattr(self.local_service, 'set_video_fps'):
                result = await self.local_service.set_video_fps(fps)
                return result
            else:
                logger.warning("Local service does not have set_video_fps method")
                return {"status": "webrtc_only", "message": f"WebRTC FPS set to {fps}, local service method not available"}
            
        except Exception as e:
            logger.error(f"Failed to set video FPS: {e}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mirror service for Squid microscope control."
    )
    parser.add_argument(
        "--cloud-service-id",
        default="mirror-microscope-control-squid-1",
        help="ID for the cloud service (default: mirror-microscope-control-squid-1)"
    )
    parser.add_argument(
        "--local-service-id",
        default="microscope-control-squid-1",
        help="ID for the local service (default: microscope-control-squid-1)"
    )
    args = parser.parse_args()

    mirror_service = MirrorMicroscopeService()
    # Override the service IDs with command line arguments
    mirror_service.cloud_service_id = args.cloud_service_id
    mirror_service.local_service_id = args.local_service_id

    loop = asyncio.get_event_loop()

    async def main():
        try:
            mirror_service.setup_task = asyncio.create_task(mirror_service.setup())
            await mirror_service.setup_task
            # Start the health check task
            asyncio.create_task(mirror_service.check_service_health())
        except Exception:
            traceback.print_exc()

    loop.create_task(main())
    loop.run_forever() 