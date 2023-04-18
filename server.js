const dotenv = require('dotenv')
const express = require('express')
const cookieParser = require('cookie-parser')
const app = express()
const cors = require('cors')
const bodyParser = require('body-parser');





app.use(cookieParser())
app.use(express.json())
app.use(bodyParser.json());
app.use(cors({
    origin:["https://malaria-frontend.onrender.com","http://44.211.176.112"]
    }))

dotenv.config({path:'./config.env'})
require('./db/conn')

app.use(require('./routes/auth'))

const PORT = process.env.PORT

app.listen(PORT, () => {
    console.log(`LISTENING ON PORT ${PORT}`)
})
