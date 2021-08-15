pip3 install --target ./package -r requirements.txt
cd package
zip -r ../activity_summary.zip .
cd ../
zip -g activity_summary.zip main.py