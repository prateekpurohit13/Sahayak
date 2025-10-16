class ErrorHandler extends Error {
    constructor(statusCode, message = "Error is here", errors = []) {
        super(message);
        this.statusCode = statusCode;
        this.errors = errors;
        this.message = message;
        this.success = false;
    }
}

export default ErrorHandler;
