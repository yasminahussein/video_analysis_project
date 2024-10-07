import os
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import plotly.io as pio
import json
from tqdm import tqdm
from io import BytesIO

# Set the frames per second (FPS) for the video
FPS = 25

heatmap_image = cv2.imread("./map_heat.png")

# Custom colorscale: transparent to yellow to red
colorscale = [
    [0, 'rgba(60, 70, 160, 0)'],   # Transparent at low values
    [0.3, 'rgba(44, 16, 157, 0.6)'], # Blue at low values
    [0.5, 'rgba(170, 200, 16, 0.7)'], # Yellow at middle values
    [0.7, 'rgba(200, 70, 16, 0.8)'], # Orange at middle values
    [1, 'rgba(255, 0, 0, 1)']    # Red at high values
]

def drawMapWithHeatmap(accumulated_points, frame_number, output_filename):
    img = heatmap_image.copy()

    Xs = [x for x, y in accumulated_points]
    Ys = [y for x, y in accumulated_points]

    map_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = go.Figure()

    # Add heatmap
    heatmap.add_trace(go.Histogram2dContour(
        x=Xs,
        y=Ys,
        colorscale=colorscale, 
        reversescale=False,
        xaxis='x',
        yaxis='y',
        contours=dict(showlabels=True, coloring='heatmap')
    ))

    # Add scatter plot to show individual points
    heatmap.add_trace(go.Scatter(
        x=Xs, 
        y=Ys,
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))

    # Overlay the heatmap on the image
    heatmap.update_layout(
        images=[go.layout.Image(
            source=Image.fromarray(map_image_rgb),
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=map_image_rgb.shape[1],
            sizey=map_image_rgb.shape[0],
            sizing="stretch",
            layer="below"
        )],
        xaxis=dict(visible=False, range=[0, map_image_rgb.shape[1]]),
        yaxis=dict(visible=False, range=[map_image_rgb.shape[0], 0], scaleanchor="x", scaleratio=1),
        showlegend=False,
        template="plotly_white"
    )

    # Convert plotly figure to an image using PIL
    img_bytes = pio.to_image(heatmap, format='png')
    img_pil = Image.open(BytesIO(img_bytes))
    
    img_pil.save(output_filename)
    print(f"Saved frame {frame_number} as {output_filename}")

def load_data_from_json(filename="frame_data.json"):
    with open(filename, "r") as json_file:
        data = json.load(json_file)
    return data

def transform_data_to_lists(frame_data_dict):
    frames = sorted(frame_data_dict.keys(), key=lambda x: int(x))
    all_projected_points = []
    all_MapIDs = []
    
    for frame in frames:
        frame_data = frame_data_dict[frame]
        projected_points = []
        MapIDs = []
        
        for ID, coords in frame_data.items():
            MapIDs.append(int(ID))
            projected_points.append((coords["x"], coords["y"]))
        
        all_projected_points.append(projected_points)
        all_MapIDs.append(MapIDs)
    
    return all_projected_points, all_MapIDs

def create_video_from_images(image_folder, output_video_file, fps):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()

def process_for_single_id(ID, all_projected_points, all_MapIDs, output_folder):
    accumulated_points = []

    # Create a folder for heatmap frames of this ID
    id_folder = os.path.join(output_folder, f"ID_{ID}")
    os.makedirs(id_folder, exist_ok=True)

    for frame_number, (projected_points, MapIDs) in tqdm(enumerate(zip(all_projected_points, all_MapIDs)), total=len(all_projected_points), desc=f"Processing ID {ID}"):
        for point, map_id in zip(projected_points, MapIDs):
            if map_id == ID:
                accumulated_points.append(point)
        output_filename = f"{id_folder}/heatmap_frame_{frame_number:04d}.png"
        drawMapWithHeatmap(accumulated_points, frame_number, output_filename)

    video_output_file = f"{output_folder}/ID_{ID}.mp4"
    create_video_from_images(id_folder, video_output_file, FPS)

# Main logic to generate heatmaps and compile them into videos
if __name__ == "__main__":
    frame_data_dict = load_data_from_json("barca_fixes.json")
    all_projected_points, all_MapIDs = transform_data_to_lists(frame_data_dict)

    output_folder = "Objects"
    os.makedirs(output_folder, exist_ok=True)

    # Get all unique IDs from the dataset
    unique_ids = set([ID for MapIDs in all_MapIDs for ID in MapIDs])

    for unique_id in unique_ids:
        process_for_single_id(unique_id, all_projected_points, all_MapIDs, output_folder)
