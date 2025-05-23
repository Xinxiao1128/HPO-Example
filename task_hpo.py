from clearml import Task
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation import DiscreteParameterRange, HyperParameterOptimizer
import logging
import os
import sys

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 ClearML 优化任务
task = Task.init(
    project_name='Agri-Pest-Detection',
    task_name='Step 4 - HPO: Train Model',
    task_type=Task.TaskTypes.optimizer
)

# 自动寻找 Step 3 任务 ID
def find_step3_task_id():
    try:
        if os.path.exists('step3_task_id.txt'):
            with open('step3_task_id.txt', 'r') as f:
                tid = f.read().strip()
                if tid:
                    logger.info(f"✔ Found Step 3 task ID from file: {tid}")
                    return tid
        tasks = Task.get_tasks(project_name='Agri-Pest-Detection', task_name='Step 3*')
        completed = [t for t in tasks if t.status in ['completed', 'published']]
        completed.sort(key=lambda t: t.data.created, reverse=True)
        return completed[0].id if completed else None
    except Exception as e:
        logger.warning(f"❌ Failed to find Step 3 task ID: {e}")
        return None

STEP2_TASK_ID = "a76d8cec491f4d75ac38d9befe59df92"
STEP3_TASK_ID = find_step3_task_id()

# 可通过参数传入 step3_id
if not STEP3_TASK_ID:
    for arg in sys.argv:
        if arg.startswith("--step3_id="):
            STEP3_TASK_ID = arg.split("=")[1]

if not STEP3_TASK_ID:
    raise ValueError("❌ Step 3 task ID not provided.")

# 配置参数
params = task.connect({
    'processed_dataset_id': STEP2_TASK_ID,
    'base_train_task_id': STEP3_TASK_ID,
    'test_queue': 'pipeline',
    'num_trials': 5,
    'time_limit_minutes': 30
})

# 超参数搜索空间
hyper_parameters = [
    DiscreteParameterRange('General/learning_rate', values=[0.0001, 0.0005, 0.001, 0.005, 0.01]),
    DiscreteParameterRange('General/batch_size', values=[16, 32, 64]),
    DiscreteParameterRange('General/weight_decay', values=[1e-6, 1e-5, 1e-4]),
    DiscreteParameterRange('General/num_epochs', values=[5, 10, 15]),
    DiscreteParameterRange('General/dropout_rate', values=[0.3, 0.4, 0.5, 0.6])
]

# 构建优化器
optimizer = HyperParameterOptimizer(
    base_task_id=params['base_train_task_id'],
    hyper_parameters=hyper_parameters,
    objective_metric_title='accuracy',
    objective_metric_series='validation',
    objective_metric_sign='max',
    optimizer_class=OptimizerOptuna,
    max_number_of_concurrent_tasks=1,
    optimization_time_limit=params['time_limit_minutes'] * 60,
    compute_time_limit=60 * 60,
    total_max_jobs=params['num_trials'],
    min_iteration_per_job=0,
    max_iteration_per_job=9999,
)

# 启动优化器
logger.info("🚀 Starting HPO optimization...")
optimizer.start_locally()
optimizer.wait()

# 获取结果
top_experiments = optimizer.get_top_experiments(top_k=min(5, params['num_trials']))
logger.info("🏁 Optimization completed. Processing results...")

if top_experiments:
    results = []
    for exp_task in top_experiments:
        try:
            metrics = exp_task.get_last_scalar_metrics()
            acc = 0.0
            if 'accuracy' in metrics and 'validation' in metrics['accuracy']:
                acc = metrics['accuracy']['validation'].get('last', 0)
            results.append({
                'id': exp_task.id,
                'name': exp_task.name,
                'accuracy': acc,
                'params': exp_task.get_parameters().get('General', {})
            })
        except Exception as e:
            logger.error(f"⚠️ Error parsing experiment {exp_task.id}: {e}")

    results.sort(key=lambda x: x['accuracy'], reverse=True)

    # 打印 top 实验
    for i, r in enumerate(results):
        logger.info(f"\n[{i+1}] {r['name']}")
        logger.info(f"Accuracy: {r['accuracy']:.4f}")
        for k, v in r['params'].items():
            logger.info(f"{k}: {v}")

    # 上传最佳实验信息
    best = results[0]
    task.upload_artifact("best_experiment_id", best['id'])
    task.upload_artifact("best_accuracy", best['accuracy'])
    task.upload_artifact("best_hyperparameters", best['params'])
    logger.info(f"\n✅ Best model: {best['accuracy']:.4f} | Task: {best['name']}")

logger.info("✅ HPO completed.")
