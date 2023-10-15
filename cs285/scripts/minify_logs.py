from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter
import os
from PIL import Image
from io import BytesIO


def extract_tensorboard_log(log_path, output_dir, image_output_path):
    # Initialize an EventAccumulator with the path to the log directory
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()  # Load the events from disk

    # Extract the scalar summaries
    scalars = {}
    for tag in event_acc.Tags()['scalars']:
        scalars[tag] = event_acc.Scalars(tag)

    # Write the scalar summaries to a new TensorBoard log
    writer = SummaryWriter(output_dir)
    for tag, events in scalars.items():
        for event in events:
            writer.add_scalar(tag, event.value, event.step)

    # Extract the last image summary
    last_image = None
    for tag in event_acc.Tags()['images']:
        image_events = event_acc.Images(tag)
        if image_events:
            last_image = image_events[-1]

            image_bytes = BytesIO(last_image.encoded_image_string)
            image = Image.open(image_bytes)
            image.save(image_output_path, save_all=True)
        else:
            print("No images")


if __name__ == '__main__':
    import argparse

    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_log_file", "-i", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    extract_tensorboard_log(
        args.input_log_file,
        args.output_dir,
        os.path.join(args.output_dir, "video.gif")
    )
