from openai import OpenAI
import base64
import os
from pathlib import Path
import argparse

class CaptioningApp:

    def __init__(self, folder_path):
        self.client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
        self.source_folder_path = folder_path
        self.destination_folder_path = folder_path


    def run_captioning(self):
        # Get folders paths
        
        try: 
            # Check the existance of folders
            os.listdir(self.destination_folder_path)
            files = os.listdir(self.source_folder_path)
            n_files = len(files)
            # Start captioning
            for i, file in enumerate(files):
                file_path = os.path.join(self.source_folder_path, file)
                destination_path = os.path.join(self.destination_folder_path, f"{Path(file_path).stem}.txt")
                img = self.load_img64(file_path)
                if not img:
                    continue

                captions = self.caption_server(img)
                if not captions:
                    print("- Failure: ", "error_message")
                    print(f"Unable to caption the image '{file}'. Check if the image is valid, with a supported format (jpeg, jpg, png, are recommended).\n")
                else:
                    with open(destination_path, "w") as f:
                        f.write(captions)
                    print(f"- Success: ", "success_message")
                    print(f"Image '{file}' has been captioned and the result saved to '{destination_path}'.\n")

            
        except Exception as e:
            print(f"Error: {e}\n", "error_message")

    def load_img64(self, file_path):
        base64_image = ""
        try:
            image = open(file_path, "rb").read()
            base64_image = base64.b64encode(image).decode("utf-8")
        except Exception as e:
            print(f"Error img: {e}\n", "error_message")
        return base64_image

    def caption_server(self, base64_image):
        completion = self.client.chat.completions.create(
            model="local-model",  # not used
            messages=[
                {
                    "role": "system",
                    "content": "This is a chat between a user and an assistant. The assistant is helping the user to caption an image.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "only ouput the response, no explanation, no sentence only words, caption this image, only main points, only tags, no full sentence, describe the mood, the background, the light, colors, all details, as much as possible, use comma between different points"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
            stream=True
        )

        captions = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                caption = chunk.choices[0].delta.content
                captions += caption
        return captions.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='folder_path', type=str, help='Images folder path')
    args = parser.parse_args()
    app = CaptioningApp(args.folder_path)
    app.run_captioning()
