import passport from "passport";
import { Strategy as LocalStrategy } from "passport-local";
import { Strategy as GoogleStrategy } from "passport-google-oauth20";
import { User } from "../models/user.model.js";

export const initializePassport = () => {
    // Local Strategy
    passport.use(
        new LocalStrategy({ usernameField: "email" }, async (email, password, done) => {
            try {
                const user = await User.findOne({ email });
                if (!user || user.authProvider !== "local") {
                    return done(null, false, { message: "User not found or not registered locally." });
                }
                const isPasswordValid = await user.isPasswordCorrect(password);
                if (!isPasswordValid) {
                    return done(null, false, { message: "Invalid credentials." });
                }
                return done(null, user);
            } catch (error) {
                return done(error);
            }
        })
    );
    // google oauth;
    passport.use(
        new GoogleStrategy(
            {
                clientID: process.env.GOOGLE_CLIENT_ID,
                clientSecret: process.env.GOOGLE_CLIENT_SECRET,
                callbackURL: "/api/auth/google/callback",
            },
            async (accessToken, refreshToken, Profiler, done) => {
                try {
                    let user = await User.findOne({ googleId: profile.id });
                    if (user) return done(null, user);

                    user = await User.findOne({ email: profile.emails[0].value });
                    if (user) {
                        user.googleId = profile.id;
                        user.authProvider = "google";
                        await user.save();
                        return done(null, user);
                    }
                    const newUser = await User.create({
                        googleId: profile.id,
                        name: profile.displayName,
                        email: profile.emails[0].value,
                        authProvider: "google",
                    });
                    return done(null, newUser);
                } catch (error) {
                    return done(error);
                }
            }
        )
    );
    passport.serializeUser((user, done) => {
        done(null, user.id);
    });
    passport.deserializeUser(async (id, done) => {
        try {
            const user = await User.findById(id);
            done(null, user);
        } catch (error) {
            done(error);
        }
    });
};
