const jwt = require('jsonwebtoken')
const User = require("../models/userSchema");

const Authenticate = async (req,res, next) => {
    try {
        //const token = req.cookies.jwtoken;
        const token = req.headers.authorization.split(" ")[1];
        console.log(token)
        const verifyToken = jwt.verify(token, process.env.SECRET_KEY);
        const { userId, username } = verifyToken;
        console.log('id'+verifyToken._id)
        const rootUser = await User.findOne({_id: userId, "tokens.token": token}); 
        console.log(rootUser)
        if(!rootUser) {
            throw new Error('User not found')
        } 
        
        req.token = token;
        req.rootUser = rootUser;
        req.userID = rootUser._id;
        req.username = username;
        next();
    } catch(e) {
        res.status(401).send('No token')
        console.log(e)
    }

}

module.exports = Authenticate
 
