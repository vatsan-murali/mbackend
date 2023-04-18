const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const userSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
  },
  phone: {
    type: String,
    required: true,
  },
  password: {
    type: String,
  },
  cpassword: {
    type: String,
  },
  tokens: [
    {
      token: {
        type: String,
        required: true,
      },
    },
  ],
  otp: { type: String },
  isVerified: {
    type: Boolean,
    default: false,
  },
});

userSchema.methods.generateAuthToken = async function () {
  try {
    const payload = {
      userId: this._id,
      username: this.username,
    };
    let token = jwt.sign(payload, process.env.SECRET_KEY);
    console.log('generated token', token);
    this.tokens = this.tokens.concat({ token: token });
    console.log('user schema', this);
    await this.save();
    console.log('saved token');
    return token;
  } catch (e) {
    console.log(e);
  }
};

const User = mongoose.model("USER", userSchema);

module.exports = User;
