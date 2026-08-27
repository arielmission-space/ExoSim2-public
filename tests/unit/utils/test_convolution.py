"""
Unit tests for the convolution utility module.

Tests the fast_convolution function which performs convolution
of images with kernels using Fourier transforms.
"""

import numpy as np

from exosim.utils.convolution import fast_convolution


class TestFastConvolution:
    """Test class for fast_convolution function."""

    def test_basic_convolution(self):
        """Test basic convolution functionality."""
        # Create a simple 2D image (16x16) with a central peak
        # Use larger dimensions to satisfy spline requirements
        im = np.zeros((16, 16))
        im[8, 8] = 1.0
        delta_im = 1.0

        # Create a simple kernel (8x8) - a 2D Gaussian-like
        x, y = np.meshgrid(np.arange(8) - 3.5, np.arange(8) - 3.5)
        ker = np.exp(-(x**2 + y**2) / 4.0)
        ker = ker / ker.sum()  # Normalize
        delta_ker = 1.0

        # Perform convolution
        result = fast_convolution(im, delta_im, ker, delta_ker)

        # Check that result is same size as image
        assert result.shape == im.shape

        # Check that result is real (no significant imaginary part)
        assert np.allclose(np.imag(result), 0, atol=1e-10)

        # Check that result has reasonable values (convolution can have small negative artifacts)
        assert np.all(np.abs(result) < 1e2)  # Reasonable magnitude

    def test_different_sampling(self):
        """Test convolution with different sampling intervals."""
        # Create image and kernel with different sampling - use larger sizes
        im = np.ones((12, 12))
        delta_im = 0.5

        ker = np.ones((8, 8)) / 64  # Normalized uniform kernel
        delta_ker = 1.0

        result = fast_convolution(im, delta_im, ker, delta_ker)

        # Check dimensions
        assert result.shape == im.shape

        # Check real output
        assert np.allclose(np.imag(result), 0, atol=1e-10)

    def test_edge_case_small_arrays(self):
        """Test with smaller but still valid arrays."""
        # Small image and kernel (minimum size for spline to work)
        im = np.ones((6, 6))
        im[3, 3] = 5.0
        ker = np.ones((6, 6)) / 36  # Normalized
        delta_im = delta_ker = 1.0

        result = fast_convolution(im, delta_im, ker, delta_ker)

        assert result.shape == im.shape
        assert np.allclose(np.imag(result), 0, atol=1e-10)

    def test_kernel_normalization_effect(self):
        """Test that kernel normalization affects output correctly."""
        # Create a uniform image - use larger size
        im = np.ones((10, 10))
        delta_im = 1.0

        # Test with two different kernel normalizations - use larger size
        ker_raw = np.ones((8, 8))
        ker_normalized = ker_raw / ker_raw.sum()
        delta_ker = 1.0

        result_raw = fast_convolution(im, delta_im, ker_raw, delta_ker)
        result_norm = fast_convolution(im, delta_im, ker_normalized, delta_ker)

        # Both should be real
        assert np.allclose(np.imag(result_raw), 0, atol=1e-10)
        assert np.allclose(np.imag(result_norm), 0, atol=1e-10)

        # Normalized kernel should give result closer to original image values
        assert np.all(np.abs(result_norm.real - 1.0) < np.abs(result_raw.real - 1.0))

    def test_symmetry_preservation(self):
        """Test that convolution preserves symmetry when appropriate."""
        # Create symmetric image - use larger size
        x, y = np.meshgrid(np.arange(-6, 7), np.arange(-6, 7))
        im = np.exp(-(x**2 + y**2) / 8.0)
        delta_im = 1.0

        # Create symmetric kernel - use larger size
        x_k, y_k = np.meshgrid(np.arange(-4, 5), np.arange(-4, 5))
        ker = np.exp(-(x_k**2 + y_k**2) / 2.0)
        ker = ker / ker.sum()
        delta_ker = 1.0

        result = fast_convolution(im, delta_im, ker, delta_ker)

        # Check symmetry (approximately, due to discrete sampling and numerical errors)
        center = result.shape[0] // 2
        assert np.allclose(result[center, :], result[center, ::-1], atol=1e-10)
        assert np.allclose(result[:, center], result[::-1, center], atol=1e-10)

    def test_linearity_property(self):
        """Test linearity property of convolution."""
        # Create two images
        im1 = np.random.rand(8, 8)
        im2 = np.random.rand(8, 8)
        alpha, beta = 2.0, 3.0
        delta_im = 1.0

        # Create kernel - use larger size
        ker = np.ones((6, 6)) / 36
        delta_ker = 1.0

        # Test linearity: conv(alpha*im1 + beta*im2) = alpha*conv(im1) + beta*conv(im2)
        combined_im = alpha * im1 + beta * im2
        result_combined = fast_convolution(combined_im, delta_im, ker, delta_ker)

        result1 = fast_convolution(im1, delta_im, ker, delta_ker)
        result2 = fast_convolution(im2, delta_im, ker, delta_ker)
        result_linear = alpha * result1 + beta * result2

        assert np.allclose(result_combined, result_linear, atol=1e-10)
