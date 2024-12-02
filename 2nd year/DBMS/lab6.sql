create database lab6;
use lab6;
create table n_rollcall(student_id int primary key, student_name varchar(50) not null, birth_date date);
create table o_rollcall(student_id int primary key, student_name varchar(50) not null, birth_date date);
insert into o_rollcall(student_id, student_name, birth_date) values
(1, 'A', '2004-06-20'),
(3, 'C', '2003-03-14');
insert into n_rollcall(student_id, student_name, birth_date) values
(1, 'A', '2004-06-20'),
(2, 'B', '2003-05-11'),
(3, 'C', '2004-01-14'),
(4, 'D', '2003-08-12'),
(5, 'E', '2005-06-21');


delimiter //
create procedure merge_rollcall_data()
begin
declare done int default false;
declare n_id int;
declare n_name varchar(50);
declare n_birth_date date;

declare n_cursor cursor for 
select student_id, student_name, birth_date from n_rollcall;

declare continue handler for not found
set done = true;

open n_cursor;

cursor_loop: loop
fetch n_cursor into n_id, n_name, n_birth_date;
if done then leave cursor_loop;
end if;

if not exists(select 1 from o_rollcall where student_id = n_id)
then insert into o_rollcall(student_id, student_name, birth_date)values (n_id, n_name, n_birth_date);
end if;
end loop;

close n_cursor;
end //
delimiter ;

call merge_rollcall_data();
select * from o_rollcall;