<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visitor Counter</title>
    <style>
        body{
            h1, p{
                text-align: center;
                font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;
            }
        }
    </style>
</head>
<body>
    <?php
    $file = 'counter.txt';
    $count = file_exists($file) ? (int)file_get_contents($file) : 0;
    file_put_contents($file, ++$count);
    ?>
    <h1>Visitor Counter</h1>
    <p>Number of Visitors: <strong> <?php echo $count; ?> </strong> </p>
</body>
</html>

