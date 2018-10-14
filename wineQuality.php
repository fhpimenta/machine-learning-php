<?php
use Phpml\CrossValidation\StratifiedRandomSplit;
use Phpml\Dataset\Demo\WineDataset;
use Phpml\Metric\Accuracy;
use Phpml\Regression\SVR;

include 'vendor/autoload.php';

$dataset = new WineDataset();

$randomSplit = new StratifiedRandomSplit($dataset, 0.3);

$classifier = new SVR();
$classifier->train($randomSplit->getTrainSamples(), $randomSplit->getTrainLabels());

$predictLabels = $classifier->predict($randomSplit->getTestSamples());

foreach ($predictLabels as &$target) {
    $target = round($target, 0);
}
$accuracy = Accuracy::score($randomSplit->getTestLabels(), $predictLabels);

echo 'Accuracy: '.Accuracy::score($randomSplit->getTestLabels(), $predictLabels);