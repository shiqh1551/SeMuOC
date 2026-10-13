class Config(object):
    def __init__(self):
        
        self.dataset = 'SWaT'

        self.input_channels = 51
        self.window_size = 128
        self.time_step = 64

        
        self.d_model = 64
        self.tcn_layers = 3
        self.attention_heads = 4
        self.dropout = 0.1
        self.score_alpha = 1.0
        self.score_beta = 1.0
        self.text_model = "intfloat/e5-base-v2"
        self.allow_hash_text_encoder = False
        self.text_embedding_cache = True
        self.text_cache_on_device = True
        self.domain_text = (
            "SWaT is a secure water treatment testbed with raw water intake, "
            "chemical dosing, filtration, reverse osmosis, and storage stages. "
            "Sensors measure flow, tank level, pressure, and conductivity. "
            "Actuators such as motorized valves and pumps control water movement."
        )
        self.variable_desc = {
            f"Channel-{idx}": f"SWaT sensor or actuator variable {idx}"
            for idx in range(self.input_channels)
        }
        self.anomaly_texts = [
            "Anomaly pattern: a valve is open but flow remains zero, indicating blockage or interruption.",
            "Anomaly pattern: tank level rapidly rises while outlet flow is low, indicating overflow risk.",
            "Anomaly pattern: a pump is active but pressure or flow does not increase.",
            "Anomaly pattern: a sensor value drifts slowly without corresponding actuator changes.",
            "Anomaly pattern: flow suddenly drops while the upstream valve remains open.",
            "Anomaly pattern: multiple related sensors show inconsistent water-treatment behavior.",
        ]

        # SemSphere one-class branch (AOC-style projection and fixed center)
        self.project_dim = 32
        self.oc_hidden_dim = None
        self.oc_pooling = "flatten"
        self.center_eps = 0.1
        self.center_normal_only = True

        
        self.use_infonce = True
        self.lambda_oc = 0.1
        self.lambda_align = 0.1
        self.lambda_anom = 0.1
        self.lambda_recon = 1.0
        self.anomaly_similarity_margin = 0.2

        
        self.num_epoch = 5
        self.early_stop = 20
        self.save_diagnostics = True
        self.keep_checkpoints = True

        
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.weight = 5e-4

        
        self.batch_size = 128
        self.drop_last = False

        
        self.detect_nu = 0.0005
        self.threshold_determine = 'floating'
