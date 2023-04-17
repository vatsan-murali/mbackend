const mongoose = require("mongoose");

const imageSchema = new mongoose.Schema({
  image: {
    type: String,
  },
  predicted: {
    type: String,
  },
  actual: {
    type: String,
  },
  id: {
    type: String,
    unique: true,
  },
  username: {
    type: String,
  },
});

function generateId() {
  const date = new Date();
  const randomNum = Math.floor(Math.random() * 100000);
  return `prediction${date.getTime()}${randomNum}`;
}

// set the id field using the custom id generator function
imageSchema.pre("save", function (next) {
  if (!this.id) {
    this.id = generateId();
  }
  next();
});

const Image = mongoose.model("IMAGE", imageSchema);

module.exports = Image;
