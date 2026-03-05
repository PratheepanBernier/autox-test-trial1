import pytest
from unittest.mock import patch, MagicMock, mock_open
import QR_code_generator

# ──────────────────────────────────────────
# Tests for: generate_qr_code
# ──────────────────────────────────────────

@patch("QR_code_generator.qrcode.make")
@patch("QR_code_generator.os.path.exists")
@patch("QR_code_generator.os.remove")
def test_generate_qr_code_happy_path(mock_remove, mock_exists, mock_make):
    # Setup
    mock_img = MagicMock()
    mock_make.return_value = mock_img
    mock_exists.return_value = False

    data = "https://example.com"
    filename = "test_qr.png"

    # Call
    result = QR_code_generator.generate_qr_code(data, filename)

    # Assert
    mock_make.assert_called_once_with(data)
    mock_img.save.assert_called_once_with(filename)
    assert result == filename

@patch("QR_code_generator.qrcode.make")
@patch("QR_code_generator.os.path.exists")
@patch("QR_code_generator.os.remove")
def test_generate_qr_code_overwrite_existing(mock_remove, mock_exists, mock_make):
    mock_img = MagicMock()
    mock_make.return_value = mock_img
    mock_exists.return_value = True

    data = "test"
    filename = "existing.png"

    result = QR_code_generator.generate_qr_code(data, filename)

    mock_remove.assert_called_once_with(filename)
    mock_make.assert_called_once_with(data)
    mock_img.save.assert_called_once_with(filename)
    assert result == filename

@patch("QR_code_generator.qrcode.make")
@patch("QR_code_generator.os.path.exists")
@patch("QR_code_generator.os.remove")
def test_generate_qr_code_empty_data(mock_remove, mock_exists, mock_make):
    mock_img = MagicMock()
    mock_make.return_value = mock_img
    mock_exists.return_value = False

    data = ""
    filename = "empty.png"

    result = QR_code_generator.generate_qr_code(data, filename)

    mock_make.assert_called_once_with(data)
    mock_img.save.assert_called_once_with(filename)
    assert result == filename

@patch("QR_code_generator.qrcode.make")
@patch("QR_code_generator.os.path.exists")
@patch("QR_code_generator.os.remove")
def test_generate_qr_code_none_data(mock_remove, mock_exists, mock_make):
    mock_exists.return_value = False
    # qrcode.make(None) will likely raise an exception
    mock_make.side_effect = TypeError("Data must not be None")

    with pytest.raises(TypeError):
        QR_code_generator.generate_qr_code(None, "none.png")

@patch("QR_code_generator.qrcode.make")
@patch("QR_code_generator.os.path.exists")
@patch("QR_code_generator.os.remove")
def test_generate_qr_code_invalid_filename(mock_remove, mock_exists, mock_make):
    mock_img = MagicMock()
    mock_make.return_value = mock_img
    mock_exists.return_value = False
    # Simulate save raising an exception
    mock_img.save.side_effect = OSError("Invalid filename")

    with pytest.raises(OSError):
        QR_code_generator.generate_qr_code("data", "")

# ──────────────────────────────────────────
# Tests for: read_qr_code
# ──────────────────────────────────────────

@patch("QR_code_generator.pyzbar.pyzbar.decode")
@patch("QR_code_generator.Image.open")
def test_read_qr_code_happy_path(mock_open, mock_decode):
    mock_img = MagicMock()
    mock_open.return_value = mock_img
    mock_decoded = [MagicMock()]
    mock_decoded[0].data = b"hello world"
    mock_decode.return_value = mock_decoded

    filename = "qr.png"
    result = QR_code_generator.read_qr_code(filename)

    mock_open.assert_called_once_with(filename)
    mock_decode.assert_called_once_with(mock_img)
    assert result == "hello world"

@patch("QR_code_generator.pyzbar.pyzbar.decode")
@patch("QR_code_generator.Image.open")
def test_read_qr_code_no_qr_found(mock_open, mock_decode):
    mock_img = MagicMock()
    mock_open.return_value = mock_img
    mock_decode.return_value = []

    filename = "no_qr.png"
    result = QR_code_generator.read_qr_code(filename)

    mock_open.assert_called_once_with(filename)
    mock_decode.assert_called_once_with(mock_img)
    assert result is None

@patch("QR_code_generator.pyzbar.pyzbar.decode")
@patch("QR_code_generator.Image.open")
def test_read_qr_code_invalid_file(mock_open, mock_decode):
    mock_open.side_effect = FileNotFoundError("File not found")
    filename = "missing.png"

    with pytest.raises(FileNotFoundError):
        QR_code_generator.read_qr_code(filename)

@patch("QR_code_generator.pyzbar.pyzbar.decode")
@patch("QR_code_generator.Image.open")
def test_read_qr_code_none_filename(mock_open, mock_decode):
    with pytest.raises(TypeError):
        QR_code_generator.read_qr_code(None)

# ──────────────────────────────────────────
# Tests for: delete_qr_code
# ──────────────────────────────────────────

@patch("QR_code_generator.os.path.exists")
@patch("QR_code_generator.os.remove")
def test_delete_qr_code_happy_path(mock_remove, mock_exists):
    mock_exists.return_value = True
    filename = "qr.png"
    result = QR_code_generator.delete_qr_code(filename)
    mock_exists.assert_called_once_with(filename)
    mock_remove.assert_called_once_with(filename)
    assert result is True

@patch("QR_code_generator.os.path.exists")
@patch("QR_code_generator.os.remove")
def test_delete_qr_code_file_not_exist(mock_remove, mock_exists):
    mock_exists.return_value = False
    filename = "notfound.png"
    result = QR_code_generator.delete_qr_code(filename)
    mock_exists.assert_called_once_with(filename)
    mock_remove.assert_not_called()
    assert result is False

@patch("QR_code_generator.os.path.exists")
@patch("QR_code_generator.os.remove")
def test_delete_qr_code_none_filename(mock_remove, mock_exists):
    mock_exists.side_effect = TypeError("filename must be str, not NoneType")
    with pytest.raises(TypeError):
        QR_code_generator.delete_qr_code(None)

@patch("QR_code_generator.os.path.exists")
@patch("QR_code_generator.os.remove")
def test_delete_qr_code_remove_raises(mock_remove, mock_exists):
    mock_exists.return_value = True
    mock_remove.side_effect = PermissionError("Permission denied")
    filename = "protected.png"
    with pytest.raises(PermissionError):
        QR_code_generator.delete_qr_code(filename)
