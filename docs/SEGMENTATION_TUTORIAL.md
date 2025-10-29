# microSAM Segmentation Tutorial

## Overview

This tutorial shows how to perform automated cell segmentation on microscope experiments using the microSAM BioEngine service. Segmentation results are stored as separate OME-Zarr experiments for easy visualization and analysis.

## Prerequisites

- Completed microscope scan with experiment saved (e.g., 'my-experiment')
- Access to Hypha server with microscope service
- microSAM service running on agent-lens workspace

## Basic Workflow

### 1. Connect to Microscope Service

```python
import asyncio
from hypha_rpc import connect_to_server, login

async def main():
    # Connect to Hypha server
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": await login({"server_url": "https://hypha.aicell.io"}),
    })
    
    # Get microscope service
    microscope = await server.get_service("your-workspace/microscope-control-squid-1")
```

### 2. Start Segmentation

```python
    # Start segmentation on all wells - single channel
    result = await microscope.segmentation_start(
        experiment_name="my-experiment",
        wells_to_segment=None,  # None = auto-detect all wells
        channel_configs=[
            {"channel": "BF LED matrix full", "min_percentile": 2.0, "max_percentile": 98.0}
        ],
        scale_level=1  # Use 1/4 resolution for faster processing
    )
    
    print(f"Segmentation started!")
    print(f"  Source: {result['source_experiment']}")
    print(f"  Target: {result['segmentation_experiment']}")
    print(f"  Wells: {result['total_wells']}")
```

### 3. Monitor Progress

```python
    # Poll status until complete
    while True:
        status = await microscope.segmentation_get_status()
        
        print(f"State: {status['state']}")
        print(f"  Progress: {status['progress']['completed_wells']}/{status['progress']['total_wells']}")
        print(f"  Current well: {status['progress']['current_well']}")
        
        if status['state'] in ['completed', 'failed']:
            break
        
        await asyncio.sleep(5)  # Check every 5 seconds
    
    if status['state'] == 'completed':
        print("✅ Segmentation completed successfully!")
    else:
        print(f"❌ Segmentation failed: {status['error_message']}")
```

### 4. Visualize Results

```python
    # Retrieve segmentation mask for a specific well
    segmentation_result = await microscope.get_single_well_region(
        well="A1",
        channel="Segmentation",  # Segmentation channel
        scale=1,
        experiment_name="my-experiment-segmentation"
    )
    
    if segmentation_result['success']:
        import base64
        import io
        from PIL import Image
        
        # Decode base64 PNG
        img_data = base64.b64decode(segmentation_result['data'])
        img = Image.open(io.BytesIO(img_data))
        
        print(f"Segmentation shape: {segmentation_result['shape']}")
        print(f"Well: {segmentation_result['well']}")
```

## Advanced Usage

### Segment Specific Wells Only

```python
# Segment only wells A1, A2, B1
result = await microscope.segmentation_start(
    experiment_name="my-experiment",
    wells_to_segment=["A1", "A2", "B1"],
    channel_configs=[
        {"channel": "BF LED matrix full"}
    ]
)
```

### Use Full Resolution

```python
# Use scale_level=0 for full resolution (slower but more accurate)
result = await microscope.segmentation_start(
    experiment_name="my-experiment",
    scale_level=0  # Full resolution
)
```

### Segment Fluorescence Channels

```python
# Segment fluorescence channel instead of brightfield
result = await microscope.segmentation_start(
    experiment_name="my-experiment",
    source_channel="Fluorescence 488 nm Ex"
)
```

### Adjust Contrast for Better Segmentation

```python
# Custom contrast adjustment for better cell boundary detection
result = await microscope.segmentation_start(
    experiment_name="my-experiment",
    min_contrast_percentile=5.0,   # Clip bottom 5% of intensities
    max_contrast_percentile=95.0   # Clip top 5% of intensities
)

# Default auto-contrast (1st-99th percentile)
result = await microscope.segmentation_start(
    experiment_name="my-experiment"
    # Uses default: min_contrast_percentile=1.0, max_contrast_percentile=99.0
)
```

### Cancel Running Segmentation

```python
# If segmentation takes too long, cancel it
result = await microscope.segmentation_cancel()
print(f"Cancelled: {result['message']}")
```

## Complete Example

```python
import asyncio
from hypha_rpc import connect_to_server, login

async def segment_experiment():
    # 1. Connect to services
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": await login({"server_url": "https://hypha.aicell.io"}),
    })
    microscope = await server.get_service("your-workspace/microscope-control-squid-1")
    
    # 2. Start segmentation
    result = await microscope.segmentation_start(
        experiment_name="my-experiment",
        scale_level=1  # 1/4 resolution for speed
    )
    print(f"✅ Started: {result['total_wells']} wells")
    
    # 3. Monitor progress
    while True:
        status = await microscope.segmentation_get_status()
        if status['state'] == 'completed':
            print(f"✅ Done! {status['progress']['completed_wells']} wells segmented")
            break
        elif status['state'] == 'failed':
            print(f"❌ Failed: {status['error_message']}")
            break
        
        print(f"⏳ Progress: {status['progress']['completed_wells']}/{status['progress']['total_wells']}")
        await asyncio.sleep(5)
    
    # 4. View results
    result = await microscope.get_single_well_region(
        well="A1",
        channel="Segmentation",
        experiment_name="my-experiment-segmentation"
    )
    print(f"📊 Segmentation shape: {result['shape']}")

# Run the workflow
asyncio.run(segment_experiment())
```

## Key Points

1. **Separate Experiments**: Segmentation results are stored in `{experiment_name}-segmentation` folder
2. **OME-Zarr Compliant**: Results follow OME-Zarr format for compatibility
3. **Instance Segmentation**: Each cell has a unique ID (uint16: 0-65535)
4. **Channel Name**: Segmentation masks are stored in "Segmentation" channel
5. **Async Operation**: Segmentation runs in background, monitor with `segmentation_get_status()`
6. **HTTP Connection**: Uses standard HTTP connection to microSAM service

## Troubleshooting

**Segmentation already running**
- Use `segmentation_cancel()` to stop current segmentation first

**No wells found**
- Ensure experiment name is correct
- Check that experiment has well zarr filesets

**microSAM connection failed**
- Check network connectivity to Hypha server
- Verify microSAM service is running on agent-lens workspace

**Out of memory**
- Use higher `scale_level` (1 or 2) for lower resolution processing
- Segment fewer wells at a time

