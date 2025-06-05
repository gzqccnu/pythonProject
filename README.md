### python project
<br>

> Attention: this repository only used to load the course work.
>         not for other use

<br>

### My work
- *src*
  contains the source code.
  and many of it are useless, the useful are under the folader `api` and `data_strong`.
  - *src/api*
    this folder contains the api of the model.
  - *src/data_strong*
    this folder contains the source code of data strong about <b>rain</b> and <b>hazy</b>
  
- *ultralytics\cfg\models\v8*

​		contains actual model configuration.

<ul>
    <ul>
        <li><i>data.yaml</i></li>
        this yaml file configures the <b>dataset destination</b>.
    </ul>
</ul>

<ul>
    <ul>
        <li><i>yolov8-MobileViT.yaml</i></li>
        this yaml file configures the <b>net structure</b> of the model.
    </ul>
</ul>

- *ultralytics\nn*

  contains the basic  component of yolo

  - *task.py*

    modify the **parse_model** function to support the modification of the net structure.

  - *modules/block.py*

    add the realization of **MobileViT**.

  - *modules/__init__.py*

    add the statement of **MobileViT** to it.
<br>

*Reference*
<br>

> this work refers the open source project `ultralytics`
