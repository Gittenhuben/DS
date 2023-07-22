# Определение возраста покупателей по фотографии<br>(Data Science, Deep Learning, Computer Vision)

[ipynb](https://github.com/Gittenhuben/Portfolio/blob/main/Age%20Recognition%20(CV)/AgeRecognition.ipynb)

## Задание

Построить модель для определения возраста человека по фотографии.

## Стек

* Python
* Pandas
* PIL
* **TensorFlow**
* Seaborn
* Matplotlib

## Итоги

Для решения задачи определения возраста по фото была выбрана нейросетевая модель ResNet50 с алгоритмом обучения Adam (learning_rate=0.0001) и функцией потерь MSE.<br>
Количество эпох было выбрано равным 100, что оказалось избыточным. Оптимальное значение: 80.<br>
Результат: **MAE на тестовой выборке 5.7526**
