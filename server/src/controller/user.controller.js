import { User } from "../models/user.model.js";
import ErrorHandler from "../utils/ErrorHandler.js";
import ErrorWrapper from "../utils/ErrorWrapper.js";

const registerUser = ErrorWrapper(async (req, res) => {
    const { name, email, password } = req.body;
    if ([name, email, password].some((field) => field?.trim() == "")) {
        throw new ErrorHandler(400, "All fields are required");
    }
    const existingUser = await User.findOne({ email });
    if (existingUser) {
        throw new ErrorHandler(400, "User already exists");
    }
    const user = await User.create({ name, email, password });
    return res.status(201).json({ message: "User registered sucessfully", userId: user._id });
});

const loginUser = ErrorWrapper(async (req, res) => {
    const { email, password } = req.body;
    const user = await User.findOne({ email });

    if (!user) {
        throw new ErrorHandler(400, "User not found");
    }
    const isPasswordValid = await user.isPasswordCorrect(password);
    if (!isPasswordValid) {
        throw new ApiError(401, "Invalid user credentials");
    }
    const accessToken = user.generateAccessToken();
    const loggedInUser = await User.findById(user._id).select("-password");
    const options = {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
    };
    return res
        .status(200)
        .cookie("accessToken", accessToken, options)
        .json({ message: "Login successful", user: loggedInUser, accessToken });
});

export { registerUser, loginUser };
