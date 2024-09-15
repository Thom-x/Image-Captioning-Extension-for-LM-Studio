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
                    print(f"Caption: '{captions}'")
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
                    "content": """
Write a four sentence caption for this image. In the first sentence describe the style and type (painting, photo, etc) of the image. Describe in the remaining sentences the contents and composition of the image. Only use language that would be used to prompt a text to image model. Do not include usage. Comma separate keywords rather than using "or". Precise composition is important. Avoid phrases like "conveys a sense of" and "capturing the", just use the terms themselves.

Good examples are:

"Photo of an alien woman with a glowing halo standing on top of a mountain, wearing a white robe and silver mask in the futuristic style with futuristic design, sky background, soft lighting, dynamic pose, a sense of future technology, a science fiction movie scene rendered in the Unreal Engine."

"A scene from the cartoon series Masters of the Universe depicts Man-At-Arms wearing a gray helmet and gray armor with red gloves. He is holding an iron bar above his head while looking down on Orko, a pink blob character. Orko is sitting behind Man-At-Arms facing left on a chair. Both characters are standing near each other, with Orko inside a yellow chestplate over a blue shirt and black pants. The scene is drawn in the style of the Masters of the Universe cartoon series."

"An emoji, digital illustration, playful, whimsical. A cartoon zombie character with green skin and tattered clothes reaches forward with two hands, they have green skin, messy hair, an open mouth and gaping teeth, one eye is half closed."
""",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Caption this image please"},
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
