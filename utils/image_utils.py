"""
Image utility functions for Googli AI Chat Module
"""
import base64
import io
from typing import Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass


# Maximum image size (5MB for Bedrock)
MAX_IMAGE_SIZE_MB = 5
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

# Maximum dimensions (prevent extremely large images)
MAX_IMAGE_DIMENSION = 4096

# Supported formats
SUPPORTED_FORMATS = {"JPEG", "PNG", "GIF", "WEBP"}


def validate_image_format(content_type: str) -> str:
    """
    Validate and extract image format from content type
    
    Args:
        content_type: MIME type of the image
        
    Returns:
        Image format (jpeg, png, gif, webp)
        
    Raises:
        ImageProcessingError: If format is not supported
    """
    format_map = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp"
    }
    
    image_format = format_map.get(content_type.lower())
    if not image_format:
        raise ImageProcessingError(
            f"Unsupported image format: {content_type}. "
            f"Supported formats: JPEG, PNG, GIF, WEBP"
        )
    
    return image_format


def resize_image_if_needed(image: Image.Image, max_dimension: int = MAX_IMAGE_DIMENSION) -> Image.Image:
    """
    Resize image if it exceeds maximum dimensions while maintaining aspect ratio
    
    Args:
        image: PIL Image object
        max_dimension: Maximum width or height
        
    Returns:
        Resized image if needed, original otherwise
    """
    width, height = image.size
    
    if width <= max_dimension and height <= max_dimension:
        return image
    
    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    
    logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def process_uploaded_image(
    image_data: bytes,
    content_type: str,
    max_size_bytes: int = MAX_IMAGE_SIZE_BYTES
) -> Tuple[str, str]:
    """
    Process uploaded image for use with Bedrock API
    
    Args:
        image_data: Raw image bytes
        content_type: MIME type of the image
        max_size_bytes: Maximum allowed size in bytes
        
    Returns:
        Tuple of (base64_encoded_data, image_format)
        
    Raises:
        ImageProcessingError: If image processing fails
    """
    # Validate size
    if len(image_data) > max_size_bytes:
        raise ImageProcessingError(
            f"Image size ({len(image_data) / 1024 / 1024:.2f}MB) exceeds "
            f"maximum allowed size ({max_size_bytes / 1024 / 1024}MB)"
        )
    
    # Validate format
    image_format = validate_image_format(content_type)
    
    try:
        # Open and validate image
        image = Image.open(io.BytesIO(image_data))
        
        # Verify format
        pil_format = image.format
        if pil_format not in SUPPORTED_FORMATS:
            raise ImageProcessingError(f"Unsupported PIL format: {pil_format}")
        
        logger.info(f"Processing image: format={image_format}, size={image.size}, mode={image.mode}")
        
        # Resize if needed
        image = resize_image_if_needed(image)
        
        # Convert to RGB if necessary (Bedrock prefers RGB)
        if image.mode not in ("RGB", "RGBA"):
            logger.info(f"Converting image from {image.mode} to RGB")
            if image.mode == "P":
                # Palette mode - convert with transparency support
                image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
            else:
                image = image.convert("RGB")
        
        # For JPEG, ensure RGB (not RGBA)
        if image_format == "jpeg" and image.mode == "RGBA":
            logger.info("Converting RGBA to RGB for JPEG")
            # Create white background
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        
        # Save to bytes with optimization
        output = io.BytesIO()
        save_format = pil_format if pil_format in SUPPORTED_FORMATS else "JPEG"
        
        if save_format == "JPEG":
            image.save(output, format=save_format, quality=85, optimize=True)
        elif save_format == "PNG":
            image.save(output, format=save_format, optimize=True)
        else:
            image.save(output, format=save_format)
        
        processed_data = output.getvalue()
        
        # Check final size
        if len(processed_data) > max_size_bytes:
            # If still too large, try more aggressive compression
            output = io.BytesIO()
            if save_format == "JPEG":
                image.save(output, format=save_format, quality=70, optimize=True)
                processed_data = output.getvalue()
            
            if len(processed_data) > max_size_bytes:
                raise ImageProcessingError(
                    f"Image too large even after compression: {len(processed_data) / 1024 / 1024:.2f}MB"
                )
        
        # Encode to base64
        base64_data = base64.b64encode(processed_data).decode('utf-8')
        
        logger.info(
            f"Image processed successfully: original={len(image_data)/1024:.1f}KB, "
            f"processed={len(processed_data)/1024:.1f}KB, "
            f"base64={len(base64_data)/1024:.1f}KB"
        )
        
        return base64_data, image_format
        
    except Exception as e:
        if isinstance(e, ImageProcessingError):
            raise
        logger.error(f"Failed to process image: {e}")
        raise ImageProcessingError(f"Failed to process image: {str(e)}")


def validate_image_file(file_size: int, content_type: str) -> None:
    """
    Quick validation of image file before processing
    
    Args:
        file_size: Size of file in bytes
        content_type: MIME type
        
    Raises:
        ImageProcessingError: If validation fails
    """
    if file_size > MAX_IMAGE_SIZE_BYTES:
        raise ImageProcessingError(
            f"Image file too large: {file_size / 1024 / 1024:.2f}MB. "
            f"Maximum allowed: {MAX_IMAGE_SIZE_MB}MB"
        )
    
    validate_image_format(content_type)


