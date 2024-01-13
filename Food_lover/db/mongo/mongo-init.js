use admin;
db.createUser({
  user: "guanjie",
  pwd: "guanjie", // Use a more secure password in production
  roles: [{ role: "readWrite", db: "fdlover" }]
});

db = db.getSiblingDB('fdlover');

db.createCollection("breakfast");
db.createCollection("lunch");
db.createCollection("dinner");
db.createCollection("snack");

db.breakfast.files.createIndex({"uploadDate": 1}, { unique: true });
db.lunch.files.createIndex({"uploadDate": 1}, { unique: true });
db.dinner.files.createIndex({"uploadDate": 1}, { unique: true });
db.snack.files.createIndex({"uploadDate": 1}, { unique: true });
