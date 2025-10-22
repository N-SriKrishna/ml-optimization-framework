"""
Download YOLOv8 model for optimization
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import logger


def download_yolov8():
    """Download YOLOv8 nano model"""
    
    logger.info("=" * 80)
    logger.info("DOWNLOADING YOLOV8 MODEL")
    logger.info("=" * 80)
    
    try:
        # Try to use ultralytics
        import torch
        from ultralytics import YOLO
        
        logger.info("\nDownloading YOLOv8n (nano) model...")
        
        # Download YOLOv8n
        model = YOLO('models/yolov8n.pt')
        
        # Get model path
        model_path = Path('models/yolov8n.pt')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to our models directory
        import shutil
        weights_path = Path.home() / '.config/Ultralytics/weights/yolov8n.pt'
        
        if weights_path.exists():
            shutil.copy(weights_path, model_path)
            logger.info(f"✓ Model saved to: {model_path}")
        else:
            # Model was downloaded but not found in expected location
            # Try to export it
            model_path = Path('yolov8n.pt')
            if model_path.exists():
                logger.info(f"✓ Model available at: {model_path}")
        
        # Get model info
        logger.info(f"\n📊 Model Information:")
        logger.info(f"  Name: YOLOv8n (Nano)")
        logger.info(f"  Task: Object Detection")
        logger.info(f"  Input Size: 640x640")
        logger.info(f"  Classes: 80 (COCO dataset)")
        
        # Try to get model size
        try:
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Model Size: {size_mb:.2f} MB")
        except:
            pass
        
        logger.info("\n✓ YOLOv8 model ready for optimization!")
        return model_path
        
    except ImportError:
        logger.error("ultralytics package not installed!")
        logger.info("\nPlease install with:")
        logger.info("  pip install ultralytics")
        return None
    
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None


def main():
    """Main function"""
    model_path = download_yolov8()
    
    if model_path:
        print("\n" + "=" * 80)
        print("Next steps:")
        print("  1. Run: python examples/real_world/02_optimize_yolov8.py")
        print("  2. This will optimize YOLOv8 with multiple techniques")
        print("  3. View results and comparison report")
        print("=" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print("ERROR: Could not download YOLOv8")
        print("Please install: pip install ultralytics")
        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
