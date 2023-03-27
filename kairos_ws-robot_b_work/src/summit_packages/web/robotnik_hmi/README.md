# robotnik_hmi
Web interface for robot management, based on php technology.


## Installation
### LAMP:
[How to install LAMP on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-linux-apache-mysql-php-lamp-stack-ubuntu-18-04)
[How to install phpMyAdmin (optional)](https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-phpmyadmin-on-ubuntu-18-04)

### Import struct of data base
The struct used in the DDBB is in the folder web/bbdd/db_robotnik.sql. Import the file in the database server, using the following steps:
* Open your database in phpMyAdmin (http://127.0.0.1/phpmyadmin/). Using the user root and the password R0b0tn1K. in case the user does not exist, create it. [How create a new user using phpMyAdmin](http://webvaultwiki.com.au/Default.aspx?Page=Create-Mysql-Database-User-Phpmyadmin&NS=&AspxAutoDetectCookieSupport=1)

* Click in New database in the left-menu.
* Click Import
* Choose the file (it is in the package rb_robomark_manager_data in the folder bbdd) and click Go

If phpMyAdmin doesn't install and only it installed MySqlServer, use the following command:

```
mysql -u root -p < [sql file path]
```

In the case the root user does not exist, create it. [How crate a new user using MySQL](https://linuxize.com/post/how-to-create-mysql-user-accounts-and-grant-privileges/)

## Add page in the web server

```
$cd /var/www/html
$ln -sf [package path]/web/robotnik_hmi robotnik_hmi
```


## Start robotnik_hmi
roslaunch robotnik_hmi robotnik_hmi.launch

## Dependencies

- ros-kinetic-rosbridge
- ros-kinetic-tf2-web-republisher
