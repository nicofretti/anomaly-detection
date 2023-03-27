-- phpMyAdmin SQL Dump
-- version 4.5.4.1deb2ubuntu2.1
-- http://www.phpmyadmin.net
--
-- Servidor: localhost:3306
-- Tiempo de generación: 12-09-2019 a las 10:47:27
-- Versión del servidor: 5.7.27-0ubuntu0.16.04.1
-- Versión de PHP: 7.0.33-0ubuntu0.16.04.5

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de datos: `db_robotnik`
--
CREATE DATABASE IF NOT EXISTS `db_robotnik` DEFAULT CHARACTER SET latin1 COLLATE latin1_swedish_ci;
USE `db_robotnik`;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `users`
--
-- Creación: 11-09-2019 a las 16:38:39
-- Última actualización: 11-09-2019 a las 17:09:45
--

DROP TABLE IF EXISTS `users`;
CREATE TABLE `users` (
  `usr_id` int(11) NOT NULL,
  `usr_username` varchar(20) NOT NULL,
  `usr_name` varchar(20) NOT NULL,
  `usr_password` varchar(20) NOT NULL,
  `usr_admin` int(11) NOT NULL DEFAULT '0',
  `usr_develop` int(11) NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- RELACIONES PARA LA TABLA `users`:
--

--
-- Volcado de datos para la tabla `users`
--

INSERT INTO `users` (`usr_id`, `usr_username`, `usr_name`, `usr_password`, `usr_admin`, `usr_develop`) VALUES
(1, 'root', 'Robotnik', 'R0b0tn1K', 1, 1),
(2, 'user', 'user1', 'user1', 0, 0),
(3, 'admin', 'admin', '4dm1N', 1, 0);

--
-- Índices para tablas volcadas
--

--
-- Indices de la tabla `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`usr_id`);

--
-- AUTO_INCREMENT de las tablas volcadas
--

--
-- AUTO_INCREMENT de la tabla `users`
--
ALTER TABLE `users`
  MODIFY `usr_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
