# How to use Jupyter lab

1. Enter `pipenv` shell:

   ```bash
   pipenv shell
   ```

   This will bring up a terminal in your virtualenv like this:

   ```py
   (my-virtualenv-name) bash-4.4$
   ```

   In that shell do:

   ```py
   python -m ipykernel install --user --name=<your-kernel-name>
   ```

   Launch jupyter notebook:

   ```py
   jupyter lab
   ```

   You should see your kernel when you create a notebook:

   ![Kernel Demo](img/kernel-demo.jpeg)

2. Next time you can just run:

   ```bash
   pipenv run jupyter lab
   ```

   instead of entering the shell and type jupyter lab.

