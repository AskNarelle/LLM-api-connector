echo "deb http://ftp.debian.org/debian stable main" >> /etc/apt/sources.list

# Update package list
apt update

# Install sqlite3
apt install -y sqlite3

#python3 app.py
gunicorn --bind=0.0.0.0 --timeout 600 app:app