detector = dict(
    # Path to the detection model
    model="/checkpoints/oad/detector/yolox_s_hdet_vga_60e_onnxonly_1280x800.onnx",
    size=(800, 1280),  # Input shape for inference
)

tracking = dict(
    track_thresh=0.5,  # Tracking confidence threshold
    match_thresh=0.8,  # Matching threshold for tracking
    track_buffer=30,  # The frames for keep lost tracks
    min_box_area=10,  # Filter out tiny boxes
)
centercrop = dict(
    # Use 0 for None(keep ratio), -1 for full dimention, float for ratio, (int,int) for size, int for square
    crop_size=(0.7, 640),  # H, W for center crop
    offset=(-100, 100),  # offset from center
)
action = dict(
    # Path to the action model
    model='/checkpoints/oad/example/trainoverfit_repvgg_oad.onnx',
    super_image_size=(3, 3),  # Super image grid size
    # Small image size, also for resize super image
    small_image_size=(224, 224),
    threshold=0.8,  # Threshold for action model
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
)

# we need class name to draw bounding boxes
class_name = ['call', 'read', 'purchase', 'chat']

# choosing device to run model
device = 'gpu'  # 'cpu'
