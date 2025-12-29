ssh -T git@github.com

git init 

git add .

git commit -m "First commit"

git branch -M main

git remote add origin git@github.com:zhanghong24/ORION.git

git remote set-url origin git@github.com:zhanghong24/ORION.git

git push -u origin main
