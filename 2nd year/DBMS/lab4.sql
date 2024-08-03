create database lab4;
use lab4;
create table customer(id int primary key, name varchar(50) not null, age int not null, address varchar(50) not null, salary decimal(10, 2) not null);
insert into customer(id, name, age, address, salary) values
(1, 'A', 24, 'Udupi', 20000.00),
(2, 'B', 24, 'Udupi', 30000.00),
(3, 'C', 24, 'Udupi', 35000.00),
(4, 'D', 24, 'Udupi', 50000.00),
(5, 'E', 24, 'Udupi', 40000.00);
select * from customer;

delimiter //
create trigger after_insert_sal_diff
after insert on customer
for each row
begin
set @my_sal_diff = concat('salary inserted is ', new.salary);
end;//
delimiter ;

delimiter //
create trigger after_update_sal_diff
after update on customer
for each row
begin
declare old_salary decimal(10, 2);
declare new_salary decimal(10, 2);
set old_salary = old.salary;
set new_salary = new.salary;
set @my_sal_diff2 = concat('Salary difference after update is ', new.salary-old.salary);
end;//
delimiter ;

delimiter //
create trigger after_delete_sal_diff
after delete on customer
for each row
begin
set @my_sal_diff3 = concat("Salary deleted is ", old.salary);
end;//
delimiter ;

-- testing of trigger
insert into customer(id, name, age, address, salary) values (6, 'F', 35, 'Mangaluru', 50000.00);
select @my_sal_diff as Sal_Diff;

-- testing of trigger
update customer set salary = 55000.00 where id = 1;
select @my_sal_diff2 as Sal_Diff;

-- testing of trigger
delete from customer where id = 6;
select @my_sal_diff3 as sal_diff;
