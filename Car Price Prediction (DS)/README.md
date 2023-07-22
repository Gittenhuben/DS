# Определение стоимости автомобилей<br>(Data Science, Machine Learning)

[ipynb](https://github.com/Gittenhuben/Portfolio/blob/main/Car%20Price%20Prediction%20(DS)/CarPricePrediction.ipynb)

## Задание

На основе исторических данных построить модель для быстрого определения рыночной стоимости автомобиля.

## Стек

* Python
* Pandas
* Numpy
* **Sklearn**
* **LightGBM**
* **CatBoost**
* Seaborn
* Matplotlib

## Итоги

Для решения задачи были использованы следующие модели:

* Случайный лес (RandomForestRegressor)
* Модели градиентного бустинга:
    * GradientBoostingRegressor
    * LGBMRegressor
    * CatBoostRegressor
    * CatBoostRegressor (с использованием встроенного обработчика категориальных признаков)

Итоговые результаты приведены в следующей таблице.

<table align=left>
    <tr>
        <th><div style="text-align:center;">Модель</div></th>
        <th><div style="text-align:center;">Параметры</div></th>
        <th><div style="text-align:center;">RMSE</div></th>
        <th><div style="text-align:center;">Время обучения, с</div></th>
        <th><div style="text-align:center;">Время предсказания одного значения, мс</div></th>
    </tr>
    <tr>
        <td><div style="text-align:left;">RandomForestRegressor</div></td>
        <td><div style="text-align:center;">max_depth: 19</div></td>
        <td><div style="text-align:center;">1507</div></td>
        <td><div style="text-align:center;"><b>69</b></div></td>
        <td><div style="text-align:center;">9.4</div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;"><b>GradientBoostingRegressor</b></div></td>
        <td><div style="text-align:center;">max_depth: 11</div></td>
        <td><div style="text-align:center;"><b>1467</b></div></td>
        <td><div style="text-align:center;">92</div></td>
        <td><div style="text-align:center;">3.6</div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;">LGBMRegressor</div></td>
        <td><div style="text-align:center;">learning_rate: 0.5<br>max_depth: 16</div></td>
        <td><div style="text-align:center;">1524</div></td>
        <td><div style="text-align:center;">355</div></td>
        <td><div style="text-align:center;">60.0</div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;">CatBoostRegressor</div></td>
        <td><div style="text-align:center;">max_depth: 16<br>iterations: 100<br>learning_rate: 0.25</div></td>
        <td><div style="text-align:center;">1480</div></td>
        <td><div style="text-align:center;">176</div></td>
        <td><div style="text-align:center;">4.8</div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;">CatBoostRegressor (cat_features)</div></td>
        <td><div style="text-align:center;">max_depth: 16<br>iterations: 100<br>learning_rate: 0.25</div></td>
        <td><div style="text-align:center;">1488</div></td>
        <td><div style="text-align:center;">157</div></td>
        <td><div style="text-align:center;"><b>0.9</b></div></td>
    </tr>
</table>



Лучшее качество предсказания - у модели GradientBoostingRegressor:<br>
RMSE: 1467 (на тестовой выборке: 1445).<br>
Среднее время предсказания одного значения: 3.6 мс.<br>
Время обучения сравнимо с наилучшим и составляет 92 с.<br>

Быстрее прочих обучается модель случайного леса: за 69 с.<br>
Лучшее время предсказания - у модели CatBoostRegressor с использованием встроенного обработчика категориальных признаков: 0.9 мс.<br>

Модель CatBoostRegressor при использовании встроенного обработчика категориальных признаков обучается и предсказывает быстрее, чем без него, но качество предсказания при этом незначительно ухудшается: RMSE увеличивается с 1480 до 1488.
