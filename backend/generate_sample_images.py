import numpy as np
import cv2
from PIL import Image
import os

def generate_lunar_surface(width=800, height=600, num_craters=15, num_boulders=30):
    """Generate a synthetic lunar surface image with craters and boulders"""
    
    # Create base surface with noise
    surface = np.random.normal(128, 20, (height, width)).astype(np.uint8)
    
    # Apply Gaussian blur for smoother surface
    surface = cv2.GaussianBlur(surface, (9, 9), 2)
    
    # Add craters
    for _ in range(num_craters):
        cx = np.random.randint(50, width - 50)
        cy = np.random.randint(50, height - 50)
        radius = np.random.randint(20, 80)
        
        # Create crater depression
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Crater profile (bowl-shaped)
        crater_mask = dist < radius
        crater_depth = np.where(crater_mask, 
                               (1 - (dist / radius)**2) * 50,
                               0)
        
        surface = np.clip(surface - crater_depth.astype(np.uint8), 0, 255).astype(np.uint8)
        
        # Add rim highlight
        rim_mask = (dist > radius * 0.8) & (dist < radius * 1.2)
        surface[rim_mask] = np.clip(surface[rim_mask] + 30, 0, 255)
    
    # Add boulders
    for _ in range(num_boulders):
        bx = np.random.randint(10, width - 10)
        by = np.random.randint(10, height - 10)
        boulder_size = np.random.randint(5, 20)
        
        # Draw boulder (bright spot)
        cv2.circle(surface, (bx, by), boulder_size, 
                  np.random.randint(180, 220), -1)
        
        # Add shadow
        shadow_offset = int(boulder_size * 0.7)
        cv2.circle(surface, (bx + shadow_offset, by + shadow_offset), 
                  boulder_size, np.random.randint(60, 90), -1)
    
    # Add surface texture
    texture = np.random.normal(0, 5, (height, width))
    surface = np.clip(surface + texture, 0, 255).astype(np.uint8)
    
    return surface

def generate_landslide_features(surface, num_landslides=3):
    """Add landslide features to the surface"""
    height, width = surface.shape
    
    for _ in range(num_landslides):
        # Landslide starting point
        start_x = np.random.randint(width // 4, 3 * width // 4)
        start_y = np.random.randint(50, height // 3)
        
        # Landslide path (curved line)
        num_points = 20
        points = []
        x, y = start_x, start_y
        
        for i in range(num_points):
            points.append([x, y])
            # Move downward with some lateral movement
            x += np.random.randint(-10, 10)
            y += np.random.randint(10, 25)
            
            if y >= height - 50:
                break
        
        points = np.array(points, dtype=np.int32)
        
        # Draw landslide path
        for i in range(len(points) - 1):
            thickness = np.random.randint(15, 30)
            color = np.random.randint(90, 120)
            cv2.line(surface, tuple(points[i]), tuple(points[i+1]), 
                    color, thickness)
        
        # Add debris at the end
        if len(points) > 0:
            debris_x, debris_y = points[-1]
            cv2.ellipse(surface, (debris_x, debris_y), 
                       (40, 25), np.random.randint(0, 180), 
                       0, 360, np.random.randint(100, 130), -1)
    
    return surface

def save_sample_images():
    """Generate and save sample lunar images"""
    os.makedirs('../data/boulders', exist_ok=True)
    os.makedirs('../data/landslides', exist_ok=True)
    
    # Generate boulder-focused images
    for i in range(3):
        surface = generate_lunar_surface(num_craters=5, num_boulders=40)
        img = Image.fromarray(surface, mode='L')
        img.save(f'../data/boulders/sample_boulder_{i+1}.png')
        print(f"Generated boulder sample {i+1}")
    
    # Generate landslide-focused images
    for i in range(3):
        surface = generate_lunar_surface(num_craters=10, num_boulders=10)
        surface = generate_landslide_features(surface, num_landslides=4)
        img = Image.fromarray(surface, mode='L')
        img.save(f'../data/landslides/sample_landslide_{i+1}.png')
        print(f"Generated landslide sample {i+1}")
    
    # Generate mixed feature image
    surface = generate_lunar_surface(num_craters=8, num_boulders=25)
    surface = generate_landslide_features(surface, num_landslides=2)
    img = Image.fromarray(surface, mode='L')
    img.save('../data/mixed_features.png')
    print("Generated mixed features sample")

if __name__ == "__main__":
    save_sample_images()
    print("\nSample images generated successfully!")
    print("You can find them in the data/ directory") 