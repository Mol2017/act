import re
import cv2
import h5py
import base64
import numpy as np
import matplotlib.pyplot as plt

from openai import OpenAI

MODEL = "gpt-4o" # "gpt-4"


def load_tensor_from_hdf5(hdf5_file_path, field_name):
    """
    Load video tensor from hdf5 file.
    """
    with h5py.File(hdf5_file_path, 'r+') as f:
        dataset = f[field_name]
        video_tensor = np.array(dataset)
    return video_tensor


def convert_frames_to_base64(tensor):
    """
    Convert image frames from numpy array to base64.

    Parameters:
    tensor: numpy array, shape [num_frames, height, width, channels]
    """
    base64_frames = []
    for frame in tensor:
        _, buffer = cv2.imencode('.jpg', frame)
        base64_frame = base64.b64encode(buffer).decode('utf-8')
        base64_frames.append(base64_frame)
    return base64_frames


def plot_tensor(tensor, index=0):
    """
    Plot an image from the tensor.

    Parameters:
    tensor: numpy array, shape [num_frames, height, width, channels]
    index: int, index of the image to plot
    """
    plt.imshow(tensor[index])
    plt.title('Random Image from Tensor')
    plt.axis('off')  # Turn off the axis numbers and ticks
    plt.show()


def interact_with_chatgpt(base64_frames):
    client = OpenAI(api_key='*')

    prompt_message = [
        {
            "role": "user",
            "content": [
                "These are frames from a video that I want to upload. Analyze what is the bimanual robot doing.",
                *map(lambda x: {"image": x, "resize": 768}, base64_frames[0::20]),
            ],
        },
        {
            "role": "user",
            "content": "The video includes the following action series: pick up the object and transfer the object to another arm.",
        },
        {
            "role": "user",
            "content": "I want you to segment the frames based on actions and specify the end frame index for each action. The end frame index should be the last frame of the action and less than the total number of frames.",
        },
        {
            "role": "user",
            "content": "The response should be in this exact format without additional words: \"action: start_frame_index - end_frame_index\"",
        },
    ]

    params = {
        "model": MODEL,
        "messages": prompt_message,
        "temperature": 0.7,
    }


    # print(PROMPT_MESSAGES)
    response = client.chat.completions.create(**params)
    print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()


def compute_intents(input_string):
    # Regular expression to find numbers
    pattern = r"(\d+)\s*-\s*(\d+)"

    # Find all matches
    matches = re.findall(pattern, input_string)

    # Extract start and end numbers
    for match in matches:
        start_number = int(match[0])
        end_number = int(match[1])
        print(f"Start number: {start_number}, End number: {end_number}")
    
    intent_tensor = np.full((400,), -1)
    value_to_set = 0

    # Set the values in the specified range to the certain value

    for start, end in matches:
        start = max(int(start) * 20, 0)
        end = min((int(end) + 1) * 20, 400)
        intent_tensor[start:end] = value_to_set  # +1 to include the end index
        value_to_set += 1

    # Print the tensor
    return intent_tensor


def write_tensor_to_hdf5(file_path, field_name, tensor):
    """
    Write tensor to hdf5 file.
    """
    with h5py.File(file_path, 'a') as h5file:
        if field_name in h5file:
            del h5file[field_name]
        h5file.create_dataset(field_name, data=tensor)


# Main function
def main():
    # Load video tensors
    file_path = "/home/wentao/aloha_data_3/episode_46.hdf5"
    field_name = "observations/images/angle"
    images = load_tensor_from_hdf5(file_path, field_name)
    
    # Convert video frames to Base64
    base64_frames = convert_frames_to_base64(images)
    # print(len(base64_frames[0::20]))

    # plot_tensor(images, 180)
    # # Interact with GPT-4-vision

    response = interact_with_chatgpt(base64_frames)

    intent_tensor = compute_intents(response)
    
    print(intent_tensor)
    write_tensor_to_hdf5("/home/wentao/aloha_data_3/episode_0.hdf5", "intent", intent_tensor)


    # # Print the response
    print("GPT-4-vision Response:")
    print(response)

if __name__ == "__main__":
    main()