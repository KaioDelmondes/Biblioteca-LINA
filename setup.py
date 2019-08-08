from setuptools import setup, find_packages

setup(name='pathology',
	version='0.1',
	url='https://github.com/KaioDelmondes/Biblioteca-LINA/tree/teste_empacotament',
	license='MIT',
	author='Kaio D.',
	author_email='kaioregodearaujo@hotmail.com',
	description='A library to learnin the process of building it on Python',
	packages=find_packages(exclude=['tests']),
	long_description=open('README.md').read(),
	zip_safe=False)