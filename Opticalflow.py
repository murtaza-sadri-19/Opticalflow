import cv2
import numpy as np


def grid_optical_flow():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to grab first frame")
        return

    # Get frame dimensions
    height, width = first_frame.shape[:2]

    # Define grid size
    grid_size_x = 25  # width of each grid cell
    grid_size_y = 25  # height of each grid cell

    # Calculate number of grid cells
    grid_cols = width // grid_size_x
    grid_rows = height // grid_size_y

    # Convert to grayscale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Parameters for Farneback optical flow
    flow_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    try:
        while True:
            # Read new frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **flow_params)

            # Create a copy of the frame for visualization
            vis_frame = frame.copy()

            # Initialize arrays to store velocity components for each grid cell
            vx_grid = np.zeros((grid_rows, grid_cols))
            vy_grid = np.zeros((grid_rows, grid_cols))

            # Process each grid cell
            for row in range(grid_rows):
                for col in range(grid_cols):
                    # Calculate pixel coordinates for grid cell
                    x1 = col * grid_size_x
                    y1 = row * grid_size_y
                    x2 = x1 + grid_size_x
                    y2 = y1 + grid_size_y

                    # Ensure we don't exceed frame boundaries
                    x2 = min(x2, width)
                    y2 = min(y2, height)

                    # Extract flow vectors for this grid cell
                    cell_flow = flow[y1:y2, x1:x2]

                    # Calculate average flow vector for the cell
                    if cell_flow.size > 0:
                        mean_flow = np.mean(cell_flow, axis=(0, 1))
                        vx, vy = mean_flow

                        # Store in grid arrays
                        vx_grid[row, col] = vx
                        vy_grid[row, col] = vy

                        # Calculate center point of grid cell for drawing
                        center_x = x1 + grid_size_x // 2
                        center_y = y1 + grid_size_y // 2

                        # Draw grid cell
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (64, 64, 64), 1)

                        # Draw flow arrow (scaled for visibility)
                        arrow_scale = 5
                        end_x = int(center_x + vx * arrow_scale)
                        end_y = int(center_y + vy * arrow_scale)

                        # Calculate magnitude for color intensity
                        magnitude = np.sqrt(vx ** 2 + vy ** 2)

                        # Only draw arrows if there's significant motion
                        if magnitude > 0.5:
                            # Color based on direction: red for horizontal, green for vertical
                            color = (0, 0, 255)  # default red for horizontal
                            if abs(vy) > abs(vx):
                                color = (0, 255, 0)  # green for vertical

                            # Draw the arrow
                            cv2.arrowedLine(vis_frame, (center_x, center_y), (end_x, end_y), color, 2)

                            # Display velocity values
                            cv2.putText(vis_frame, f"{vx:.1f},{vy:.1f}",
                                        (center_x - 20, center_y + 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Display the result
            cv2.imshow('Grid Optical Flow', vis_frame)

            # Update previous frame
            prev_gray = gray.copy()

            # Exit on ESC key
            k = cv2.waitKey(30) & 0xff
            if k == 27:  # ESC key
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Return the last calculated velocity grids
        return vx_grid, vy_grid


if __name__ == "__main__":
    grid_optical_flow()