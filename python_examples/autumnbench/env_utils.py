import json
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import List, Dict, Any, Tuple
from .interpreter_module import Interpreter
import yaml
from generated.mara import mara_environment_pb2 as env_pb2

DEFAULT_COLOR = "black"

# Define the color mapping for different grid entities
COLOR_MAP = {
    "gray": "gray",
    "gold": "gold",
    "green": "green",
    "mediumpurple": "mediumpurple",
    "purple": "purple",
    "white": "white",
    "yellow": "yellow",
    "blue": "blue",
    "red": "red",
    "orange": "orange",
    # Add more mappings as needed
}


def load_yaml_to_dict(filepath):
    """
    Loads a YAML file and returns its content as a Python dictionary.

    Args:
        filepath (str): The path to the YAML file.

    Returns:
        dict: A dictionary representing the YAML content, or None if an error occurs.
    """
    try:
        with open(filepath, 'r') as file:
            # Load the YAML data
            data = yaml.safe_load(file)
            # The PyYAML library directly returns a dictionary if the YAML structure is a mapping.
            # For your specific YAML, keys are integers and values are strings.
            # We can ensure keys are integers if they are not already (though PyYAML usually handles this).
            if isinstance(data, dict):
                return {int(k): str(v) for k, v in data.items()}
            else:
                print(
                    f"Error: YAML content in '{filepath}' is not a dictionary."
                )
                return None
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{filepath}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def get_action_space_interactive(
        grid_size: int,
        time_step: int = 1,
        truncated_action_space: bool = True) -> List[env_pb2.Action]:
    acts = []
    if time_step > 0:
        acts = ["left", "right", "up", "down"]
        if truncated_action_space:
            acts.append(f"click [0-{grid_size-1}] [0-{grid_size-1}]")
        else:
            for i in range(grid_size):
                for j in range(grid_size):
                    acts.append(f"click {i} {j}")
    acts.append("noop")
    return [env_pb2.Action(text_data=act) for act in acts]


def interpreter_action_to_text(interpreter: Interpreter, action: str) -> str:
    if action.startswith("click"):
        x, y = action.strip().split()[1:]
        interpreter.click(int(x), int(y))
        return True
    elif action == "left":
        interpreter.left()
        return True
    elif action == "right":
        interpreter.right()
        return True
    elif action == "up":
        interpreter.up()
        return True
    elif action == "down":
        interpreter.down()
        return True
    elif action == "noop":
        return True
    return False


def parse_grid(render_output: str):
    """
    Parses the JSON string output from render_all() into a grid and its size.

    Args:
        render_output (str): The JSON string representation of the grid.

    Returns:
        tuple: A tuple containing the grid dictionary and grid size.
    """
    try:
        elem_dict = json.loads(render_output)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return {}, 0
    grid = elem_dict
    grid_size = elem_dict.pop("GRID_SIZE", 0)
    return grid, grid_size


def render_grid(grid: Dict[str, Any]) -> str:
    """
    Renders the grid into a string representation.

    Args:
        grid (Dict[str, Any]): The grid dictionary.
    
    """
    grid_size = grid.pop("GRID_SIZE", 0)
    grid_matrix = [[DEFAULT_COLOR for _ in range(grid_size)]
                   for _ in range(grid_size)]
    for elem in grid:
        for subelem in grid[elem]:
            col_idx = subelem["position"]["x"]
            row_idx = subelem["position"]["y"]
            color_key = subelem["color"].lower()
            color = COLOR_MAP.get(color_key, color_key)
            if (row_idx >= 0 and row_idx
                    < grid_size) and (col_idx >= 0 and col_idx < grid_size):
                grid_matrix[row_idx][col_idx] = color
    return '\n'.join([' '.join(row) for row in grid_matrix])


def render_grid_to_matrix(grid: Dict[str, Any]) -> List[List[str]]:
    grid_size = grid.pop("GRID_SIZE", 0)
    grid_matrix = [[DEFAULT_COLOR for _ in range(grid_size)]
                   for _ in range(grid_size)]
    for elem in grid:
        for subelem in grid[elem]:
            col_idx = subelem["position"]["x"]
            row_idx = subelem["position"]["y"]
            color_key = subelem["color"].lower()
            color = COLOR_MAP.get(color_key, color_key)
            if (row_idx >= 0 and row_idx
                    < grid_size) and (col_idx >= 0 and col_idx < grid_size):
                grid_matrix[row_idx][col_idx] = color
    return grid_matrix


def check_grid_same(grid1: List[List[str]], grid2: List[List[str]],
                    inv_mask: List[List[bool]]) -> bool:
    for i in range(len(grid1)):
        for j in range(len(grid1[0])):
            if not inv_mask[i][j]:
                continue
            if grid1[i][j] != grid2[i][j]:
                return False
    return True


def render_grid_matplotlib(grid: Dict[str, Any],
                           output_path: str = None, background_color: str = "black") -> str:
    """
    Renders the grid into a base64 encoded image string using matplotlib.
    Optionally saves the image to a file if output_path is provided.

    Args:
        grid (Dict[str, Any]): The grid dictionary.
        output_path (str, optional): The path to save the image. Defaults to None.

    Returns:
        str: Base64 encoded string of the rendered JPEG image.
    """
    grid_size = grid.get("GRID_SIZE", 0)
    img_buffer = io.BytesIO()

    fig, ax = plt.subplots()
    if grid_size == 0:
        # Handle empty or invalid grid
        ax.text(0.5, 0.5, "No grid data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Create a numerical matrix for imshow
        grid_matrix_numeric = np.zeros((grid_size, grid_size), dtype=int)
        
        # Temporary grid to extract colors and positions
        temp_grid_matrix = [[background_color for _ in range(grid_size)]
                            for _ in range(grid_size)]

        # Populate temp_grid_matrix with actual colors from the grid data
        processed_grid = grid.copy()
        grid_size = processed_grid.pop("GRID_SIZE", None)

        local_color_list = [background_color]

        for elem in processed_grid:
            for subelem in processed_grid[elem]:
                col_idx = subelem["position"]["x"]
                row_idx = subelem["position"]["y"]
                if (row_idx >= 0 and row_idx < grid_size) and (col_idx >= 0 and col_idx < grid_size):
                    color_key = subelem["color"].lower()
                    actual_color = COLOR_MAP.get(color_key, color_key)
                    # If color is an integer, convert it to the actual color name
                    if isinstance(actual_color, int):
                        actual_color = COLOR_DICT.get(actual_color, color_key)
                    temp_grid_matrix[row_idx][col_idx] = actual_color
                    if actual_color not in local_color_list:
                        local_color_list.append(actual_color)

        color_to_int = {color: i for i, color in enumerate(local_color_list)}

        for r in range(grid_size):
            for c in range(grid_size):
                grid_matrix_numeric[r,
                                    c] = color_to_int[temp_grid_matrix[r][c]]

        valid_mpl_colors = []
        for color_name in local_color_list:
            # Handle mask color specially
            if color_name == "mask":
                valid_mpl_colors.append("slategrey")
            else:
                try:
                    mcolors.to_rgb(color_name)
                    valid_mpl_colors.append(color_name)
                except ValueError:
                    valid_mpl_colors.append(background_color)

        cmap = mcolors.ListedColormap(valid_mpl_colors)
        norm = mcolors.BoundaryNorm(np.arange(-0.5, len(valid_mpl_colors)),
                                    cmap.N)

        # Use constant figure size and dynamically size cells to fit
        plt.close(fig)  # Close the initial empty fig, ax
        fig, ax = plt.subplots(figsize=(5, 5))  # Fixed figure size

        # Calculate cell size and spacing based on grid_size to fit in fixed figure
        total_space = 1.0  # Total space available for the grid (normalized)
        margin = 0.01  # Smaller margin around the entire grid
        gap_size = 0.01  # Fixed gap size between cells

        # Calculate available space after margins
        available_space = total_space - 2 * margin

        # Calculate space taken by gaps (grid_size - 1 gaps between cells)
        total_gap_space = (grid_size - 1) * gap_size

        # Remaining space for actual cells
        cell_space = available_space - total_gap_space
        cell_size = cell_space / grid_size

        for r in range(grid_size):
            for c in range(grid_size):
                color_idx = grid_matrix_numeric[r, c]
                color = valid_mpl_colors[color_idx]

                # Calculate position with consistent spacing
                x_pos = margin + c * (cell_size + gap_size)
                y_pos = margin + r * (cell_size + gap_size)

                # Create rectangle for this cell
                rect = plt.Rectangle((x_pos, y_pos),
                                     cell_size,
                                     cell_size,
                                     facecolor=color,
                                     edgecolor='black',
                                     linewidth=0.5)
                ax.add_patch(rect)

        # Set the limits and aspect ratio to fit the entire grid
        ax.set_xlim(0, total_space)
        ax.set_ylim(0, total_space)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert y-axis to match typical grid orientation

        ax.set_xticks([])
        ax.set_yticks([])

    # Save to buffer for base64 encoding
    plt.savefig(img_buffer, format='jpeg')

    # Save to file if output_path is provided
    if output_path:
        plt.savefig(
            output_path
        )  # Matplotlib will infer format from extension or default to PNG

    plt.close(fig)
    img_buffer.seek(0)
    image_bytes = img_buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')


def render_string_grid_matplotlib(grid: List[List[str]],
                                  output_path: str = None) -> str:
    """
    Renders a grid structured as list of lists of strings into a base64 encoded image string using matplotlib.
    Entries marked as "mask" will be displayed as slategrey.
    Optionally saves the image to a file if output_path is provided.

    Args:
        grid (List[List[str]]): The grid as a list of lists of strings (color names).
        output_path (str, optional): The path to save the image. Defaults to None.

    Returns:
        str: Base64 encoded string of the rendered JPEG image.
    """
    grid = [row.split(" ") for row in grid.split('\n')]
    if not grid or not grid[0]:
        # Handle empty grid
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "No grid data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='jpeg')
        plt.close(fig)
        img_buffer.seek(0)
        image_bytes = img_buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    num_rows = len(grid)
    num_cols = len(grid[0]) if grid else 0
    img_buffer = io.BytesIO()

    # Use constant figure size and dynamically size cells to fit
    fig, ax = plt.subplots(figsize=(5, 5))  # Fixed figure size

    # Calculate cell size and spacing based on grid dimensions to fit in fixed figure
    total_space = 1.0  # Total space available for the grid (normalized)
    margin = 0.01  # Smaller margin around the entire grid
    gap_size = 0.01  # Fixed gap size between cells

    # Calculate available space after margins
    available_space = total_space - 2 * margin

    # Calculate space taken by gaps and cell sizes for both dimensions
    total_gap_space_x = (num_cols - 1) * gap_size if num_cols > 1 else 0
    total_gap_space_y = (num_rows - 1) * gap_size if num_rows > 1 else 0

    # Remaining space for actual cells
    cell_space_x = available_space - total_gap_space_x
    cell_space_y = available_space - total_gap_space_y

    # Calculate cell sizes for both dimensions
    cell_width = cell_space_x / num_cols if num_cols > 0 else 0
    cell_height = cell_space_y / num_rows if num_rows > 0 else 0

    # Use the smaller dimension to maintain square cells
    cell_size = min(cell_width, cell_height)

    for r in range(num_rows):
        for c in range(len(grid[r])):
            color_name = grid[r][c].lower()

            # Handle mask entries
            if color_name == "mask":
                color = "slategrey"
            else:
                # Try to get color from COLOR_MAP, fallback to the color name itself
                color = COLOR_MAP.get(color_name, color_name)

            # Validate color for matplotlib
            try:
                mcolors.to_rgb(color)
                actual_color = color
            except ValueError:
                actual_color = DEFAULT_COLOR

            # Calculate position with consistent spacing
            x_pos = margin + c * (cell_size + gap_size)
            y_pos = margin + r * (cell_size + gap_size)

            # Create rectangle for this cell
            rect = plt.Rectangle((x_pos, y_pos),
                                 cell_size,
                                 cell_size,
                                 facecolor=actual_color,
                                 edgecolor='black',
                                 linewidth=0.5)
            ax.add_patch(rect)

    # Set the limits and aspect ratio to fit the entire grid
    ax.set_xlim(0, total_space)
    ax.set_ylim(0, total_space)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match typical grid orientation

    ax.set_xticks([])
    ax.set_yticks([])

    # Save to buffer for base64 encoding
    plt.savefig(img_buffer, format='jpeg')

    # Save to file if output_path is provided
    if output_path:
        plt.savefig(
            output_path
        )  # Matplotlib will infer format from extension or default to PNG

    plt.close(fig)
    img_buffer.seek(0)
    image_bytes = img_buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')


def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val

    # Hue calculation
    if diff == 0:
        h = 0
    elif max_val == r:
        h = 60 * ((g - b) / diff % 6)
    elif max_val == g:
        h = 60 * ((b - r) / diff + 2)
    else:
        h = 60 * ((r - g) / diff + 4)

    # Normalize hue to 0-360
    h = (h + 360) % 360

    # Saturation calculation
    s = 0 if max_val == 0 else diff / max_val

    # Value calculation
    v = max_val

    return h, s, v


def get_color_from_hsv(h: float,
                       s: float,
                       v: float,
                       alpha: float = 1.0) -> str:
    # Handle transparency
    if alpha < 0.5:
        return "transparent"

    # Handle grayscale
    if s < 0.2:
        if v < 0.2: return "black"
        if v > 0.8: return "white"
        return "gray"

    # Color wheel segments with more precise boundaries
    if h < 15 or h >= 345: return "red"
    if h < 45: return "orange"
    if h < 75: return "yellow"
    if h < 165: return "green"
    if h < 195: return "cyan"
    if h < 255: return "blue"
    if h < 285: return "purple"
    if h < 345: return "mediumpurple"
    return "red"


def get_color_name(color_data: Dict[str, float]) -> str:
    """Convert color data dictionary to color name"""
    r = color_data.get('r', 0.0)
    g = color_data.get('g', 0.0)
    b = color_data.get('b', 0.0)
    alpha = color_data.get('alpha', 1.0)

    h, s, v = rgb_to_hsv(r, g, b)
    return get_color_from_hsv(h, s, v, alpha)
