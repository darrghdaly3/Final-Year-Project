import numpy as np 

## Function that works with the heatmap to detect what facial features activate GradCam
def detect_features(heatmap):
    
    h, w = heatmap.shape
    
    ## Splitting face into 3 areas for eyes, nose and mouth
    top = heatmap[0:int(h/3), :]
    middle = heatmap[int(h/3):int(2*h/3), :]
    bottom = heatmap[int(2*h/3):h, :]

    ## Finding the area in the heatmap that flags the most
    intensity = {
        "eyes": np.mean(top),
        "nose": np.mean(middle),
        "mouth": np.mean(bottom)
    }
    
    highest = max(intensity.values())
    selected_area = [k for k, v in intensity.items() if v >= highest * 0.75]
    
    ## Creating the textual explanation
    if len(selected_area) == 1:
        return selected_area[0]
        ##return f"The detector found the {area_text} area to be possibly manipulated or fake."

    elif len(selected_area) == 2:
        return f"{selected_area[0]} and {selected_area[1]}"
    
    elif len(selected_area) == 3:
        return f"{selected_area[0]} and {selected_area[1]} and {selected_area[2]}"

    else:
        return "An error has occured! Please reupload media."