## Self-hosted Mongo DB instance

### On Ubuntu 22.04

To self-host MongoDB on Ubuntu, you need to install the MongoDB Community Server, configure it, and secure it. This involves adding the MongoDB repository, installing the necessary packages, starting the service, and potentially configuring remote access and user authentication. 
Here's a more detailed breakdown:

1. Update System Packages:
First, update your Ubuntu system's package index:

```bash
sudo apt update
```

2. Install MongoDB:
Add the MongoDB repository: Import the MongoDB GPG key and create a list file for the repository. The specific commands will depend on your Ubuntu version. For example, for Ubuntu 22.04:

```bash
wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add - 
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
```

Then, reload the package index: 

```bash
sudo apt update
```

Install MongoDB packages. 

```bash
sudo apt install -y mongodb-org
```

This will install the core MongoDB packages. 

3. Start and Enable MongoDB:

Start the MongoDB service. 

```bash
sudo systemctl start mongod
```

Enable the service to start on boot:

```bash
sudo systemctl enable mongod
```

Verify the service status. 

```bash
sudo systemctl status mongod
```

4. Configure MongoDB:

* Edit the configuration file: The main configuration file is usually located at `/etc/mongod.conf`. 

* Bind IP: By default, MongoDB only binds to the local interface (`127.0.0.1`). To allow remote connections, you need to change the bindIP setting to include the server's IP address or `0.0.0.0` to listen on all interfaces.
  
* Make sure the Authentication is disabled:

```bash
sudo vim /etc/mongod.conf

# security:
#  authorization: enabled
```
  
* Restart the service: After making changes to the configuration, restart the MongoDB service:

```bash
sudo systemctl restart mongod
```

5. Secure MongoDB: 

* Create an Admin user: Connect to the MongoDB shell using mongosh and create an Admin user with the necessary roles for your application:

```bash
mongosh
test> use admin
switched to db admin
admin> db.createUser( { user: "myUserAdmin", pwd: "xxxxx", roles: [ { role: "userAdminAnyDatabase", db: "admin" }, { role: "readWriteAnyDatabase", db: "admin" } ] } )
{ ok: 1 }
```
* Exit `mongosh`

* Enable Authentication: For enhanced security, enable authentication by setting security.authorization to enabled.

```bash
sudo vim /etc/mongod.conf

security:
   authorization: enabled
```  
* Restart the service: After making changes to the configuration, restart the MongoDB service:

```bash
sudo systemctl restart mongod
```
* Connect to the mongo DB using `mongosh` with authentication:

```bash
ubuntu@ip-x-x-x-x:~$ mongosh --port 27017  --authenticationDatabase "admin" -u "myUserAdmin" -p
Enter password: ***********
```

* Create a user with the necessary roles for your application: 

```bash
use your_database
db.createUser({user: "your_username", pwd: "your_password", roles: [ { role: "readWrite", db: "your_database" } ]})
```

* Consider using UFW: If you have UFW (Uncomplicated Firewall) enabled, you can further restrict access to your MongoDB server by allowing connections only from trusted IP addresses.

6. Connect to MongoDB:

* Use mongosh: You can connect to your MongoDB instance using mongosh from the server itself or from a remote machine if you've enabled remote access.

* Example connection string: `mongodb://your_username:your_password@your_server_ip:27017/your_database `

* connect to MongoDB from python client as:

```python
>>> MONGODB_URI = "mongodb://your_username:your_password@your_server_ip:27017/your_database"
>>> client = pymongo.MongoClient(MONGODB_URI)
>>> coll=db.data
>>> coll.insert_one({"name": "document1", "value": 10})
InsertOneResult(ObjectId('687a58fb7aa580cacfec08b5'), acknowledged=True)
```

### Note if you are deploying on AWS EC2

If you are deploying MongoDB server / cluster on AWS EC2 make sure there is an inbound rule for port 27017 in the relevant security group for the AWS EC2 instance as shown on the Figure below:

![Figure: Security Group Inbound Rule for port 27017](images/security_group_inbound_rule_port_27017.png)
