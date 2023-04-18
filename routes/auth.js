const express = require("express");
const dotenv = require("dotenv");
const router = express.Router();
const User = require("../models/userSchema");
const bodyParser = require("body-parser");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const authenticate = require("../middleware/authenticate");
const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;
const client = require("twilio")(accountSid, authToken);
const nodeMailer = require("nodemailer");
const Image = require("../models/imageSchema");
const { TokenInstance } = require("twilio/lib/rest/oauth/v1/token");
const { errorMonitor } = require("events");


// GET PAGE

router.get("/", (req,res) => {
  console.log('in get')
return res.json({mssg: "GET PAGE"})
})


// Sign Up
router.post("/register", async (req, res) => {
  const { username, phone, password, cpassword } = req.body;
  if (!password || !cpassword) {
    return res.status(422).json({ error: "Fill out credentials" });
  }

  try {
    const hashedPassword = bcrypt.hashSync(password, 12);
    const user = await User.findOneAndUpdate(
      { username, phone },
      { password: hashedPassword },
      { new: true, upsert: true }
    );

    if (password !== cpassword) {
      return res
        .status(422)
        .json({ error: "Please make sure your passwords match" });
    }
    // if(phone) {
    //   await client.messages.create({
    //     body: `Your verification code`,
    //     from: '+15855523108',
    //     to: phone,
    //   });
    // }

    const userRegister = await user.save();
    console.log(userRegister);

    return res.status(201).json({ message: "User registered successfully" });
  } catch (err) {
    console.log(err);
  }
});

// Login
router.post("/signin", async (req, res) => {
  console.log('IN LOGIN BACKEND')
  const { phone, password } = req.body;
  if (!phone || !password) {
    return res.status(422).json({ error: "Fill out credentials" });
  }

  try {
    const userCheck = await User.findOne({
      phone: phone,
    });
    console.log('userCheck',userCheck)
    if (userCheck) {
      console.log(password, userCheck.password);
      const isMatch = bcrypt.compare(password, userCheck.password);
      const token = await userCheck.generateAuthToken();
      console.log('token',token);
      res.cookie("jwtoken", token, {
        expires: new Date(Date.now() + 25892000000),
        httpOnly: true,
      });
      if (isMatch) {
        return res.status(201).json({ message: "User present!" });
      } else {
        return res.status(422).json({ error: "User does not exist!" });
      }
    } else {
      return res.status(422).json({ error: "User does not exist!" });
    }
  } catch (err) {
    console.log(err);
  }
});

// Main

router.get("/site", authenticate, (req, res) => {
  console.log("Welcome to main site");
  res.send(req.rootUser);
});

// Logout

router.get("/logout", (req, res) => {
  console.log("Welcome to logout page");
  res.clearCookie("jwtoken", { path: "/" });
  res.status(200).send("User Logout");
});

// OTP

router.post("/sendOTP", async (req, res) => {
  const { username, recipient, phone } = req.body;
  if (!username || !phone) {
    return res.status(422).json({ error: "Fill out credentials" });
  }

  const userCheck = await User.findOne({ phone });
  if (userCheck) {
    return res.status(422).json({ error: "User already exists" });
  } else {
    const otp = Math.floor(100000 + Math.random() * 900000); // Generate random 6-digit OTP
    await client.messages.create({
      body: `Your verification code is ${otp}`,
      from: "+15855523108",
      to: recipient,
    });
    try {
      const updatedUser = await User.findOneAndUpdate(
        { username, phone },
        { otp },
        { new: true, upsert: true }
      );

      if (updatedUser) {
        return res.status(200).json({ message: "OTP sent successfully" });
      } else {
        return res.status(500).json({ error: "Server error" });
      }
    } catch (err) {
      console.error(err);
      return res.status(500).json({ error: "Server error" });
    }
  }
});

// Verify
router.post("/verifyOTP", async (req, res) => {
  const { recipient, otp, phone } = req.body;
  try {
    console.log(recipient, otp);
    // Find the user by recipient number and the OTP sent to them
    const user = await User.findOne({ phone });
    console.log(user);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    if (otp !== user.otp) {
      return res.status(404).json({ error: "Invalid OTP" });
    }
    user.isVerified = true;
    const newUser = await user.save();
    console.log(newUser);
    return res.json({ message: "OTP verified successfully" });
  } catch (error) {
    console.log(error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// Send Email

router.post("/send-email", (req, res) => {
  const { recipient, username, phone } = req.body;
  const otp = Math.floor(100000 + Math.random() * 900000); // Generate random 6-digit OTP
  let transporter = nodeMailer.createTransport({
    service: "gmail",
    auth: {
      user: "bc.predict@gmail.com",
      pass: "xqfrpccrckwgcipp",
    },
    tls: {
      rejectUnauthorized: false,
    },
  });
  let mailOptions = {
    from: '"Breast Cancer Prediction" <bc.predict@gmail.com>',
    to: recipient,
    subject: "Account Creation",
    text:
      "You have recently created an account with us.\nYour Verification OTP is " +
      otp +
      "\nIf you haven't made an account please contact site administrator.",
  };

  transporter.sendMail(mailOptions, async (error, info) => {
    try {
      const updatedUser = await User.findOneAndUpdate(
        { username, phone },
        { otp },
        { new: true, upsert: true }
      );
      if (updatedUser) {
        return res.status(200).json({ message: "OTP sent successfully" });
      } else {
        return res.status(500).json({ error: "Server error" });
      }
    } catch (error) {
      console.log;
      return res
        .status(500)
        .json({ error: "An error occurred while sending the email" });
    }
  });
});

// Send image

router.post("/send-image", (req, res) => {
  const { prediction, imagePreviewUrl, choice } = req.body;
  const token = req.cookies.jwtoken;
  const decoded = jwt.verify(token, process.env.SECRET_KEY);
  const username = decoded.username;

  console.log(choice);
  const newImage = new Image({
    image: imagePreviewUrl,
    predicted: prediction,
    actual: choice,
    username: username,
  });

  newImage
    .save()
    .then(() => {
      return res.status(200).json({ message: "image saved" });
    })
    .catch((e) => {
      console.log(e);
      return res.status(422).json({ error: e });
    });
});

router.get("/get-images", async (req, res) => {
  try {
    const token = req.cookies.jwtoken;
    const decoded = jwt.verify(token, process.env.SECRET_KEY);
    const username = decoded.username;

    const images = await Image.find({ username: username });
    if (!images) {
      return res.status(404).json({ error: "No images found" });
    }
    res.status(200).json({ images });
  } catch (err) {
    console.log(errorMonitor);
    res.status(500).json({ error: "Server Error" });
  }
});

module.exports = router;
