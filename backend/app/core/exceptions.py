class FileUploadError(Exception):
    """ 
    Base exception for all document-upload failures.
    
    Catching this parent exception would catch any og the more
    specific upload exceptions declared below.
    """
    pass

class InvalidFileError(FileUploadError):
    """ 
    Raised when the uploaded file is empty or has an invalid filename.
    """
    pass


class UnsupportedFileTypeError(FileUploadError):
    """ 
    Raised when the uploaded file is not accepted pdf.
    """
    pass


class FileTooLargeError(FileUploadError):
    """ 
    Raised when the uploaded file exceeds the configured size limit.
    """

class FileStorageError(FileUploadError):
    """ 
    Raised when a validated file cannot be written to local storage.
    """
    pass
