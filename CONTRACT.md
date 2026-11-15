# 业务资料约定

所有时刻使用带时区 ISO8601。load 是学校 school 在 target_time 的一小时 kWh，用 known_at 表示首次可见或修订时点，负值非法。weather 是同一目标小时 target_time 的气象预报，issued_at 为预报发布时点，known_at 是系统实际收到时点；两个时点均不得晚于预测 origin。calendar 提供目标小时所处 school 的开课类型，且受 known_at 约束。请求 horizons 是从 origin 起严格为正的整数小时偏移。预测训练样本的标签可见截止时点由调用 origin 决定，不能用最终版标签回写旧特征。school 的 climate_zone 与 floor_area 是事先可得的静态属性。基线只提供按 origin 固定的天气与已知负荷视图，没有学习、区间、选择或模型存取。若区间为边际而非联合覆盖，须在结果中说明；未知与零负荷不能混同。
