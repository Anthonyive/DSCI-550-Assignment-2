### How to work on your version of work

Number one rule: DON'T mess up the `main` branch.

1. Be sure to pull all new stuffs from GitHub first:

   ```bash
   $ git pull origin <your-branch-name>
   ```

   If you don't have your branch first, you can replace `<your-branch-name>` with `main`.

2. Create your own branch:

   ```bash
   $ git checkout -b <your-branch-name>
   ```

   and verify your branch has successfully created:

   ```bash
   $ git branch
   ```

   you should see your newly created branch. This step should be only done once. Press `q` to quit.

3. Switch to your branch:

   ```bash
   $ git checkout <your-branch-name>
   ```

4. After you have done your work, use these commands to push your code to GitHub:

   ```bash
   # add all your changes to the staging environment:
   # '.' basically means 'all'
   $ git add .
   ```

   ```bash
   # commit your changes:
   # be sure to write a good message that summarizes your work.
   $ git commit -m "<your-message>"
   ```

   ```bash
   # push on the GitHub:
   $ git push origin <your-branch-name>
   ```

5. Finally, GitHub will ask me to merge the code. You usually don't have to worry about that. If there are too many changes and you are sure what need to merge and how to merge, you can do it yourself, but let other people know.


![haha](https://i.redd.it/qghk5de68sz21.jpg)
