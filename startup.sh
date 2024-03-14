echo "deb http://ftp.debian.org/debian stable main" >> /etc/apt/sources.list

# Update package list
apt update

# Install sqlite3
apt install -y sqlite3

#python3 app.py
uvicorn app:app --host 0.0.0.0 