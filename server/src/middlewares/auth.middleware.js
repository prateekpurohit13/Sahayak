import jwt from "jsonwebtoken";
import { User } from "../models/user.model.js";
import ErrorHandler from "./../utils/ErrorHandler.js";
import ErrorWrapper from "./../utils/ErrorWrapper.js";

const generateAccessAndRefreshTokens = async (userId) => {
    try {
        const user = await User.findById(userId);
        const accessToken = user.generateAccessToken();
        const refreshToken = user.generateRefreshToken();

        user.refreshToken = refreshToken;
        await user.save({ validateBeforeSave: false });

        return { accessToken, refreshToken };
    } catch (error) {
        throw new ErrorHandler(500, "Something went wrong while generating refresh and access tokens");
    }
};

export const verifyJWT = ErrorWrapper(async (req, res, next) => {
    try {
        const incomingAccessToken = req.cookies?.accessToken || req.header("Authorization")?.replace("Bearer ", "");

        if (!incomingAccessToken) {
            throw new ErrorHandler(401, "Unauthorized request: No token provided.");
        }

        const decodedToken = jwt.verify(incomingAccessToken, process.env.ACCESS_TOKEN_SECRET);

        const user = await User.findById(decodedToken?._id).select("-password");

        if (!user) {
            throw new ErrorHandler(401, "Invalid Access Token: User not found.");
        }

        req.user = user;
        next();
    } catch (error) {
        if (error.name === "TokenExpiredError") {
            const incomingRefreshToken = req.cookies?.refreshToken;

            if (!incomingRefreshToken) {
                throw new ErrorHandler(401, "Session expired. Please log in again.");
            }

            try {
                const decodedRefreshToken = jwt.verify(incomingRefreshToken, process.env.REFRESH_TOKEN_SECRET);
                const user = await User.findById(decodedRefreshToken?._id);

                if (!user || user.refreshToken !== incomingRefreshToken) {
                    throw new ErrorHandler(401, "Refresh token is expired or invalid. Please log in again.");
                }

                const { accessToken, refreshToken: newRefreshToken } = await generateAccessAndRefreshTokens(user._id);

                const options = { httpOnly: true, secure: true, sameSite: "Lax" };

                res.cookie("accessToken", accessToken, options);
                res.cookie("refreshToken", newRefreshToken, options);

                req.user = user;
                next();
            } catch (refreshError) {
                res.clearCookie("accessToken");
                res.clearCookie("refreshToken");
                throw new ErrorHandler(401, "Refresh token is expired or invalid. Please log in again.");
            }
        } else {
            throw new ErrorHandler(401, error?.message || "Invalid access token.");
        }
    }
});
