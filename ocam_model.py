"""
Python conversion of the omnidirectional camera model by Davide Scaramuzza.
This file replaces 'ocam_functions.h' and 'ocam_functions.cpp'.
"""

import numpy as np
import math

class OcamModel:
    def __init__(self):
        """
        Initializes the ocam_model parameters.
        """
        self.pol = []
        self.length_pol = 0
        self.invpol = []
        self.length_invpol = 0
        self.xc = 0.0
        self.yc = 0.0
        self.c = 0.0
        self.d = 0.0
        self.e = 0.0
        self.width = 0
        self.height = 0

    def _read_next_data_line(self, f):
        """ 
        Helper function to read lines until it finds one 
        that is not empty or a comment. 
        """
        while True:
            line = f.readline()
            if not line:
                # End of file
                raise EOFError("Unexpected end of file while reading calibration.")
            line = line.strip()
            # Skip empty lines and comment lines (starting with '#')
            if line and not line.startswith('#'):
                return line.split()

    def load_calib(self, filepath):
        """
        Reads the parameters of the omnidirectional camera model from a TXT file.
        This replaces 'get_ocam_model'.
        """
        try:
            with open(filepath, 'r') as f:
                
                # --- Read polynomial coefficients ---
                line_parts = self._read_next_data_line(f) # Reads the '5 ...' line
                self.length_pol = int(line_parts[0])
                pol_vals = [float(x) for x in line_parts[1:]]
                
                # Read more lines if the coefficients are split
                while len(pol_vals) < self.length_pol:
                    line_parts = self._read_next_data_line(f) # This will skip blanks/comments
                    pol_vals.extend([float(x) for x in line_parts])
                self.pol = np.array(pol_vals)

                # --- Read inverse polynomial coefficients ---
                line_parts = self._read_next_data_line(f) # Reads the '17 ...' line
                self.length_invpol = int(line_parts[0])
                invpol_vals = [float(x) for x in line_parts[1:]]

                # Read more lines if the coefficients are split
                while len(invpol_vals) < self.length_invpol:
                    line_parts = self._read_next_data_line(f)
                    invpol_vals.extend([float(x) for x in line_parts])
                self.invpol = np.array(invpol_vals)

                # --- Read center coordinates ---
                line_parts = self._read_next_data_line(f) # Reads '607.000000 967.000000'
                self.xc = float(line_parts[0])
                self.yc = float(line_parts[1])

                # --- Read affine coefficients ---
                line_parts = self._read_next_data_line(f) # Reads '1.000000 0.000000 0.000000'
                self.c = float(line_parts[0])
                self.d = float(line_parts[1])
                self.e = float(line_parts[2])

                # --- Read image size ---
                line_parts = self._read_next_data_line(f) # Reads '1216 1936'
                self.height = int(line_parts[0])
                self.width = int(line_parts[1])
                
                print(f"Calibration file '{filepath}' loaded successfully.")
                return 0

        except FileNotFoundError:
            print(f"ERROR: File {filepath} cannot be opened.")
            return -1
        except Exception as e:
            print(f"ERROR: An error occurred while parsing {filepath}: {e}")
            return -1

    def cam2world(self, point2D):
        """
        CAM2WORLD projects a 2D point onto the unit sphere.
        Input: point2D is a (N, 2) or (H, W, 2) numpy array of (row, col) coordinates.
        Output: point3D is a (N, 3) or (H, W, 3) numpy array of (X, Y, Z) vectors.
        """
        # Ensure input is a numpy array
        point2D = np.asarray(point2D)
        input_shape = point2D.shape
        # Reshape to (N, 2) for easier processing
        point2D = point2D.reshape(-1, 2)

        invdet = 1.0 / (self.c - self.d * self.e)
        
        xp = invdet * (   (point2D[:, 0] - self.xc) - self.d * (point2D[:, 1] - self.yc) )
        yp = invdet * ( -self.e * (point2D[:, 0] - self.xc) + self.c * (point2D[:, 1] - self.yc) )
        
        r = np.sqrt(xp**2 + yp**2)  # distance from center
        
        # Use numpy's polynomial evaluation
        # The polynomial is pol[0] + pol[1]*r + pol[2]*r^2 + ...
        # np.polyval expects coefficients in reverse order (highest power first)
        zp = np.polyval(self.pol[::-1], r)
        
        # Normalize to unit norm
        norm = np.sqrt(xp**2 + yp**2 + zp**2)
        
        # Handle division by zero for points at the center
        invnorm = np.zeros_like(norm)
        mask = norm > 1e-18
        invnorm[mask] = 1.0 / norm[mask]
        
        point3D = np.zeros((point2D.shape[0], 3))
        point3D[:, 0] = invnorm * xp
        point3D[:, 1] = invnorm * yp
        point3D[:, 2] = invnorm * zp

        # Reshape back to original (..., 3)
        return point3D.reshape(input_shape[:-1] + (3,))

    def world2cam(self, point3D):
        """
        WORLD2CAM projects a 3D point on to the image.
        Input: point3D is a (N, 3) or (H, W, 3) numpy array of (X, Y, Z) coordinates.
        Output: point2D is a (N, 2) or (H, W, 2) numpy array of (row, col) coordinates.
        """
        # Ensure input is a numpy array
        point3D = np.asarray(point3D)
        input_shape = point3D.shape
        # Reshape to (N, 3) for easier processing
        point3D = point3D.reshape(-1, 3)
        
        norm = np.linalg.norm(point3D[:, 0:2], axis=1)
        point2D = np.zeros((point3D.shape[0], 2))

        # Points not on the z-axis
        mask = norm > 1e-18
        
        theta = np.zeros_like(norm)
        # Use arctan2 for numerical stability
        theta[mask] = np.arctan2(point3D[mask, 2], norm[mask])
        
        # Use numpy's polynomial evaluation
        rho = np.polyval(self.invpol[::-1], theta)
        
        invnorm = np.zeros_like(norm)
        invnorm[mask] = 1.0 / norm[mask]
        
        x = point3D[:, 0] * invnorm * rho
        y = point3D[:, 1] * invnorm * rho
        
        u = x * self.c + y * self.d + self.xc
        v = x * self.e + y + self.yc
        
        point2D[:, 0] = np.where(mask, u, self.xc)
        point2D[:, 1] = np.where(mask, v, self.yc)
        
        # Reshape back to original (..., 2)
        return point2D.reshape(input_shape[:-1] + (2,))

    def create_perspective_undistortion_lut(self, out_height, out_width, sf):
        """
        Create Look Up Table for undistorting the image into a perspective image.
        This is a vectorized version of the C++ function.
        
        Args:
            out_height (int): The desired height of the undistorted image.
            out_width (int): The desired width of the undistorted image.
            sf (float): The "zoom" factor.
            
        Returns:
            mapx (np.ndarray): The CV_32FC1 map for x-coordinates (columns).
            mapy (np.ndarray): The CV_32FC1 map for y-coordinates (rows).
        """
        # Center of the new perspective image
        Nxc = out_height / 2.0
        Nyc = out_width / 2.0
        Nz = -out_width / sf  # Z-coordinate in 3D

        # Create 2D grids of (i, j) coordinates for the *output* image
        # 'ij' indexing means i (rows) goes from 0 to H-1, j (cols) from 0 to W-1
        i_coords, j_coords = np.meshgrid(np.arange(out_height), np.arange(out_width), indexing='ij')

        # Calculate 3D coordinates (X, Y, Z) for each pixel in the output image
        # In the C++ code: M[0] = (i - Nxc), M[1] = (j - Nyc)
        # This maps (row, col) to (X, Y)
        M_x = i_coords - Nxc
        M_y = j_coords - Nyc
        M_z = np.full_like(M_x, Nz)
        
        # Stack into a (H, W, 3) array
        M = np.stack([M_x, M_y, M_z], axis=-1)
        
        # Project all 3D points to 2D image points
        # m will be (H, W, 2), where m[..., 0] is 'row' and m[..., 1] is 'col'
        m = self.world2cam(M)
        
        # For cv2.remap:
        # mapx gets the source *column* (x-coordinate), which is m[..., 1]
        # mapy gets the source *row* (y-coordinate), which is m[..., 0]
        mapx = m[..., 1].astype(np.float32)
        mapy = m[..., 0].astype(np.float32)
        
        return mapx, mapy

    def create_panoramic_undistortion_lut(self, out_height, out_width, Rmin, Rmax):
        """
        Create Look Up Table for undistorting the image into a panoramic image.
        This is a vectorized version of the C++ function.

        Args:
            out_height (int): Desired height of the panoramic image.
            out_width (int): Desired width of the panoramic image.
            Rmin (float): Minimum radius to undistort.
            Rmax (float): Maximum radius to undistort.

        Returns:
            mapx (np.ndarray): The CV_32FC1 map for x-coordinates (columns).
            mapy (np.ndarray): The CV_32FC1
        """
        # Create 2D grids of (i, j) coordinates for the *output* image
        i_coords, j_coords = np.meshgrid(np.arange(out_height), np.arange(out_width), indexing='ij')
        
        theta = -(j_coords.astype(float)) / out_width * 2 * np.pi
        rho = Rmax - (Rmax - Rmin) / out_height * i_coords.astype(float)
        
        # C++: *( data_mapx + i*width+j ) = yc + rho*sin(theta);
        # C++: *( data_mapy + i*width+j ) = xc + rho*cos(theta);
        # mapx (cols) = yc + rho*sin
        # mapy (rows) = xc + rho*cos
        mapx = (self.yc + rho * np.sin(theta)).astype(np.float32)
        mapy = (self.xc + rho * np.cos(theta)).astype(np.float32)

        return mapx, mapy