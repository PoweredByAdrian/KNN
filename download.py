import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

labels_file="export.json"

# token for autorization
headers = {
    "Authorization": f"Token {os.getenv('LABEL_STUDIO_TOKEN')}"
}

if os.path.exists(labels_file):
    print(f"labels file {labels_file} already exist, not downloading it")
    with open(labels_file,"r",encoding="utf-8") as f:
        labels=json.load(f)
else:    
    labels_url = "https://label-studio.semant.cz/api/projects/16/export"

    response = requests.get(labels_url, headers=headers,verify=False)

    # Check response
    if response.status_code == 200:
        with open("export.json", "w", encoding="utf-8") as f:
            f.write(response.text)
            print(response)
        print("Export successful! Saved as export.json")
    else:
        print(f"Failed to export: {response.status_code}, {response.text}")
    labels=response.json()


# Filter function: Check if "Popis u obrázku" exists in any annotation
def has_popis_u_obrazku(task):
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") == "rectanglelabels" and "Popis u obrázku" in result.get("value", {}).get("rectanglelabels", []):
                return True
    return False

# Apply the filter
filtered_labels = [task for task in labels if has_popis_u_obrazku(task)]
# Save the filtered data
with open("filtered_export.json", "w", encoding="utf-8") as f:
    json.dump(filtered_labels, f, indent=2,ensure_ascii=False)

save_dir = "downloaded_images"
os.makedirs(save_dir, exist_ok=True)


base_url = "https://label-studio.semant.cz"
for task in filtered_labels[:10]:
    # retriev path to image
    image_path = task.get("data", {}).get("image", "")
    if image_path.startswith("/data/local-files/"):
        # prepend the base url
        image_url = base_url + image_path
        
        # Fix filename issue on Windows (replace ':' with '_')
        image_name = str(task["id"]) + ".jpg"
        save_path= os.path.join(save_dir,image_name)

        # verify false since label studio doest have valid certificate
        response = requests.get(url=image_url, headers=headers, verify=False)
        if response.status_code == 200:
            with open(save_path, "wb") as img_file:
                img_file.write(response.content)
            print(f"Downloaded: {image_path} to {image_name}")
        else:
            print(f"Failed to download: {image_url}")

print("Download complete.")
