import { Router } from "express";
import passport from "passport";
import { verifyJWT } from "./../middlewares/auth.middleware.js";
import { registerUser } from "./../controller/user.controller.js";
const router = Router();

const loginSuccessHandler = (req, res) => {
    const user = req.user;
    const accessToken = user.generateAccessToken();

    const options = { httpOnly: true, secure: true, sameSite: "Lax" };

    res.cookie("accessToken", accessToken, options).redirect("http://localhost:5173/dashboard");
};

router.post("/register", registerUser);
router.post(
    "/login",
    passport.authenticate("local", { failureRedirect: "/login-failed", session: false }),
    (req, res) => {
        // This part runs only on successful authentication
        const accessToken = req.user.generateAccessToken();
        const options = { httpOnly: true, secure: true, sameSite: "Lax" };
        res.status(200).cookie("accessToken", accessToken, options).json({
            message: "Login successful",
            user: req.user,
        });
    }
);

router.get(
    "/auth/google",
    passport.authenticate("google", {
        scope: ["profile", "email"],
    })
);

router.get(
    "/auth/google/callback",
    passport.authenticate("google", {
        failureRedirect: "http://localhost:5173/login-failed",
    }),
    loginSuccessHandler
);

router.get("/me", verifyJWT, (req, res) => {
    res.status(200).json({ user: req.user });
});

router.post("/logout", verifyJWT, (req, res) => {
    res.clearCookie("accessToken", {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "Lax",
    });
    res.status(200).json({ message: "User logged out successfully" });
});

export default router;
