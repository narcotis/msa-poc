from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Table, Enum, Date, JSON, DateTime, Float, ARRAY
from sqlalchemy.orm import relation, relationship
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm.relationships import foreign
from sqlalchemy_utils import EmailType, UUIDType

# SQLAlchemy model 생성 전에 반드시 import
from .database import Base

# 1단계 association
# class Membership(Base):
#     __tablename__ = "membership"
#     user_id = Column(Integer, ForeignKey("users.user_id"), primary_key=True)
#     project_id = Column(Integer, ForeignKey("projets.project_id", primary_key=True))
#     role = Column(Enum("Manager", "Member"))
#     user = relationship("User", back_populates="membership")
#     project = relationship("Project", back_populates="membership")


##########################################
########### PROJECT APP ################
#########################################

# 2단계 association
membership = Table('membership', Base.metadata,
                          Column('user_id', Integer, ForeignKey("users.user_id"), primary_key=True),
                          Column('project_id', Integer, ForeignKey("projects.project_id"), primary_key=True),
                          Column('role', Enum("Manager","Member"))
                          )


class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # organization nullable 은 ALAB 운영자들을 위해
    organization = Column(Integer, ForeignKey("organizations.organization_id"), index=True, nullable=True)
    username = Column(String, unique=True)
    email = Column(EmailType, unique=True)          # admin@organization.~~~
    phone_number = Column(String, default='')       # format
    real_name = Column(String, default='')          # UTF-8
    is_initiated = Column(Boolean, default=False)   # invitation pending or not
    department = Column(String, default='')
    position = Column(String, default='')
    init_date = Column(Date, nullable=True)
    sub = Column(String, unique=True)               # unique sub from keycloak
    # initiated date
    # 1단계 association
    # projects = relationship("Membership", back_populates='users')
    projects = relationship("Project", secondary=membership, back_populates='users')


class Project(Base):
    __tablename__ = "projects"
    project_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    product = Column(Integer, unique=True, index=True)  # AFK
    organization = Column(Integer, ForeignKey("organizations.organization_id"), index=True)
    license = Column(Integer, ForeignKey("licenses.license_id"), index=True)
    is_running = Column(Boolean, default= True)       # 제작중, order app 에서 form 생성시, publish

    # 1단계 association
    # users = relationship("Membership", back_populates='projects')
    users = relationship("User", secondary=membership, back_populates='projects')


class License(Base):
    __tablename__ = 'licences'
    license_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    code = Column(UUIDType(binary=False), unique=True)
    interval = Column(Enum("monthly", "yearly", "half", "quarter"))   # 예측 주기
    start_date = Column(Date)  # license 시작 날짜
    end_date = Column(Date)    # license 끝 날짜
    organization = Column(Integer, ForeignKey("organizations.organization_id"), index=True)
    product = Column(Integer, index=True, unique=True) # AFK
    description = Column(String, default='')
    is_trial = Column(Boolean, default=False)        # default false
    is_activated = Column(Boolean, default=False)   # default false
    is_last = Column(Boolean, default=True)        # 임시. ADMIN page 에서 last license 만 보여주기 위함
    project = relationship("Project", backref="license")


# Public S3, S3 bucket은 고정. 고정된 S3 bucket에 지정된 path만 적용되면 됨.
# s3://bucket_name/organization_id/filename.png
class Organization(Base):
    __tablename__ = "organizations"
    organization_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, unique=True)
    domain = Column(String, unique=True)         # 구 url
    bi = Column("FILEFIELD", default="ALAB favicon")            # public s3 path, url
    admin_user = Column(Integer, ForeignKey("users.user_id"), nullable=True, index=True)
    users = relationship("User", backref="organizations", cascade="all, delete") # cascade , 조금 더 고민
    projects = relationship("Project", backref="organizations")
    licenses = relationship("License", backref="organizations")


# Order App 의 Product 중에서 일부분만 저장.
class Product(Base):
    __tablename__ = "products"
    product_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    category = Column(Enum("HR", "Univ"))  # HR / Univ (Scope) - tag in Optimizer Experiment
    product_name = Column(String)   # 상품 이름 (퇴사예측)
    # feature_template = Column(Integer, ForeignKey("feature_templates.feature_template_id"))
    icon = Column(String)                   # product type icon (from public s3)


##########################################
########### ORDER APP ################
#########################################
# Product Metadata
class Product(Base):
    __tablename__ = "products"
    product_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    category = Column(Enum("HR", "Univ"))  # HR / Univ (Scope) - tag in Optimizer Experiment
    product_name = Column(String)   # 상품 이름 (퇴사예측)
    best_model_criteria = Column(String)  # auc, f1-score ......and so on
    coverage = Column(Enum("Full", "Half", "Demo"), default="Full")
    type = Column(Enum("classification", "regression"))
    # feature_template = Column(Integer, ForeignKey("feature_templates.feature_template_id"))
    icon = Column(String)                   # product type icon (from public s3)
    image = Column(String)                  # product image path (from public s3)
    form = relationship("Form", backref="product")
    feature_template = relationship("FeatureTemplate", backref="product")
    orders = relationship("Order", backref="product") #for shortcut
    pipeline_template = relationship("PipelineTemplate", backref="product")


class Template(Base):
    __tablename__ = "templates"
    template_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    product = Column(Integer, ForeignKey("products.product_id"), unique=True, index=True)
    # step_template = Column(Integer, ForeignKey("step_templates.step_template_id"))
    feature_template = Column(Integer, ForeignKey("feature_templates.feature_template_id"), unique=True)
    pipeline_template = Column(Integer, ForeignKey("pipeline_templates.pipeline_template_id"), unique=True)

    forms = relationship("Form", backref="template")
    step_templates = relationship("StepTemplate", backref="template")


# 새로운 탄생
class Form(Base):
    __tablename__ = "forms"
    form_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    project = Column(Integer, index=True)       # AFK
    template = Column(Integer, ForeignKey("templates.template_id"))     # template layer
    form_response = Column(Integer, ForeignKey("form_responses.form_response_id"), unique=True)


# FormResponse 는 항상 최상단에는 last 만 보이게끔
class FormResponse(Base):
    __tablename__ = "form_responses"
    form_response_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # feature added 가져오기위해서. license 는 바뀌어도 product 는 유지되어야함
    step_response = Column(JSON, nullable=True)          # 현재는 present / future 2개, 추후 변경 가능 === 분기, 월별 등등.
    step_status = Column(JSON)                           # Step Status List
    progress = Column(JSON)  # status and duration for pipeline
    feature = Column(Integer, ForeignKey("features.feature_id"), nullable=True)
    # feature 는 project 에 mapping 되기 때문에 foreignkey 만 가지는것으로.
    is_locked = Column(Boolean, default=False)        # 학습중 mutex

    # Order 가 FormResponse 를 ForeignKey 로 가짐
    # FormResponse : Order = 1 : N 가능
    orders = relationship("Order", backref="form_response")
    form = relationship("Form", backref="form_response")


# feature depends on ProjectType (Template)
class FeatureTemplate(Base):
    __tablename__ = "feature_templates"
    feature_template_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    essentials = Column(JSON)           # 필수 Feature
    options = Column(JSON)              # 선택 Feature
    formats = Column(ARRAY(String, dimensions=1, zero_indexes=True, as_tuple=False))    # for column format examples
    # feature_template = Column(JSON)  # 프로젝트 타입별로 미리 작성된 선택피쳐들 템플릿.
    # feature_added = Column(JSON, nullable=True)  # added features
    feature = relationship("Feature", backref="feature_template")
    template = relationship("Template", backref="feature_template")


# Feature depends on Project
class Feature(Base):
    __tablename__ = "features"
    feature_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    project = Column(Integer, index=True, unique=True) #AFK
    feature_template = Column(Integer, ForeignKey("feature_templates.feature_template_id"), index=True)
    feature_added = Column(JSON)        # 추가 Feature
    feature_selected = Column(JSON, nullable=True)  # List of selected template_feature (or ARRAY)


class StepTemplate(Base):
    # 이거 템플릿임.
    __tablename__ = "step_templates"
    step_template_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    sequence = Column(Integer)
    title = Column(String)
    icon = Column(Integer)      # public s3 path?
    content_title = Column(String)
    content_desc1 = Column(String, nullable=True)
    content_desc2 = Column(String, nullable=True)
    content_desc3 = Column(String, nullable=True)   # 빨간줄 두줄이기 때문에 추가
    template = Column(Integer, ForeignKey("templates.template_id"), index=True)


# FormResponse 를 기반으로 Order 작성 (산출)
# Modeling 과 Dashboard 제작에 필요한 부분
class Order(Base):
    __tablename__= "orders"
    order_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # form response 는 동일해도, 여러번 학습을 하면 여러번의 Order 발.....생
    form_response = Column(Integer, ForeignKey("form_responses.form_response_id"), index=True)
    encrypted_feature = Column(JSON, nullable=True)
    data = Column(JSON, nullable=True)                 # Training Data ID, Validation Data ID, Inference Data ID
    started_at = Column(Date, nullable=True)           # Dashboard 는 Order(Train) 기준으로 History
    finished_at = Column(Date, nullable=True)          # Dashboard finished date for History
    train = Column(Integer, unique=True, index=True, nullable=True)     # AFK. Order : Train = 1 : 1
    best_model = Column(Integer, unique=True, index=True, nullable=True)     # AFK.
    # product = Column(Integer, ForeignKey("products.product_id"), index=True)     # for shortcut
    project = Column(Integer, index=True)                       # For Solutions query, AFK
    # Data 의 JSON Format 에 따라서 다른 종류의 train 을 호출할 수 있다면? or product 에 따라서?
    # 암호화 적용됨 - encrypted_feature 을 null 인지 아닌지 판별해서 적용
    # Data metadata, Model Metric 은 각각 Data App, Optimizer 에서 받아옴
    # pipeline = relationship("Pipeline", backref="order")

    '''
    {
        "training_data": 27,
        "validation_data": 28,
        "inference_data": 30
    }
    '''


# product 별 progress 의 template
class PipelineTemplate(Base):
    __tablename__ = "pipeline_templates"
    pipeline_template_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # product = Column(Integer, ForeignKey("products.product_id"), unique=True, index=True)
    content = Column(JSON)              # 각 step 별 내용들, 말풍선 title, 말풍선 등
    template = relationship("Template", backref='pipeline_template')
    # pipeline = relationship("Pipeline", backref="order")

#
# # pipeline template 를 포함한 progress
# class Pipeline(Base):
#     __tablename__ = "pipelines"
#     pipeline_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
#     order = Column(Integer, ForeignKey("orders.order_id"), index=True)
#     pipeline_template = Column(Integer, ForeignKey("pipeline_templates.pipeline_template_id"), index=True)
#     progress = Column(JSON)             # status and duration for each pipelines


####################################################


# in optimizer
membership = Table(
    'membership', Base.metadata,
    Column('researcher_id', Integer, ForeignKey("researchers.researcher_id"), primary_key=True),
    Column('experiment_id', Integer, ForeignKey("experiments.experiment_id"), primary_key=True),
)


class Researcher(Base):
    __tablename__ = "researchers"
    researcher_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    researcher_sub = Column(String, unique=True, index=True)        # keycloak "sub"
    experiments = relationship("Experiment", secondary=membership, back_populates='researchers')


# optimizer project 와 다른 service project mapping
class Experiment(Base):
    __tablename__ = "experiments"
    experiment_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    experiment_name = Column(String)        # 외부 project_id (ex. lean_001)
    tag = Column(String)        # HR / Univ / Lean (Scope)
    researchers = relationship("Researcher", secondary=membership, back_populates='experiments')


# Data App
class UploadedDataset(Base):
    __tablename__ = "uploaded_datasets"
    uploaded_dataset_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # 기존의 project
    experiment = Column(Integer, ForeignKey("experiments.experiment_id"), cascade="all")
    # cascade all / delete / delete_orphan
    file_path = Column(String)
    file_size = Column(String)
    encoding = Column(String)           # UTF-8, CP949 ...
    label = Column(String)
    dataset_type = Column(String, nullable=True)        # training set, validation set, inference set ...
    description = Column(String, nullable=True)         # 2017년 ~ 2019년 3분기 + 2019년 4분기 (퇴사여부 데이터 포함)
    created_at = Column(DateTime)                       # Data Version
    updated_at = Column(DateTime)


class DatasetMetadata(Base):
    __tablename__ = "dataset_metadatas"
    dataset_metadata_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    uploaded_dataset = Column(Integer, ForeignKey("uploaded_datasets.uploaded_dataset_id"))
    row_count = Column(Integer)     # data record 수
    missing_rate = Column(JSON)     # 각 column 별 결측치 비율
    data_formats = Column(JSON)     # 각 column 별 datatype 및 format


## optimizer web
class Train(Base):
    __tablename__ = "trains"
    train_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    experiment = Column(Integer, ForeignKey("experiments.experiment_id"))
    # models.ForeignKey(ProjectTable, on_delete=models.PROTECT, related_name='project_train')
    dag_run_id = Column(String)
    # datafile = Column(Integer, ForeignKey("uploaded_datasets.upload_dataset_id"))   # AFK
    data_files = Column(JSON)  # 데이터 AFK 여러개 저장.
    # models.ForeignKey(UploadedDataset, on_delete=models.PROTECT, related_name='uploaded_file')
    label = Column(String)
    type = Column(Integer, ForeignKey("model_types.model_type_id"))
    # models.ForeignKey(ModelType, models.CASCADE, related_name='train_type', blank=True, null=True)
    description = Column(String)  # AutoML 인지, SKLearn( = Manual 로 하나만...) 인지.
    ratio = Column(Float, default=0.3)
    sampling = Column(Enum("STD", "OVER", "UNDER"), default="STD")
    coverage = Column(Enum("FULL", "HALF", "DEMO"), default="FULL")
    # is_finished = models.BooleanField(null=True, default=None)
    is_finished = Column(Boolean, default=False)
    is_failed = Column(Boolean, default=False)


class SplitDataset(Base):
    __tablename__= "split_datasets"
    split_dataset_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    train = Column(Integer, ForeignKey("trains.train_id"))
    train_set = Column(String)          # s3 file path
    # models.FileField(upload_to=datetime.datetime.today().strftime("%Y-%m-%d") + '/train_set/', blank=True, null=True)
    # valid_set = models.FileField(upload_to=datetime.datetime.today().strftime("%Y-%m-%d") + '/validation_set/', blank=True, null=True)
    test_set = Column(String)


class TrainEncoder(Base):
    __tablename__= "train_encoders"
    train_encoder_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    train = Column(Integer, ForeignKey("trains.train_id"))
    train_encoder_pkl = Column(String)          # s3 encoder.pkl path


class TrainScaler(Base):
    __tablename__= "train_scalers"
    train_scaler_id = "train_scalers"
    train = Column(Integer, ForeignKey("trains.train_id"))
    train_scaler_pkl = Column(String)           # s3 scaler.pkl path


class ModelType(Base):
    __tablename__= "model_types"
    model_type_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_type = Column(String)     # Regression / Classification / Recommendation ...
    autoML = Column(ARRAY(String, dimensions=1, zero_indexes=True, as_tuple=True))
    sklearn = Column(ARRAY(String, dimensions=1, zero_indexes=True, as_tuple=True))
    metrics = Column(ARRAY(String, dimensions=1, zero_indexes=True, as_tuple=True))


class ModelService(Base):
    __tablename__ = "model_services"
    model_service_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    url = Column(String)        # model dns
    is_available = Column(Boolean, default=False)       # 서빙 상태인지 아닌지 (inference 가능한지)
    is_locked = Column(Boolean, default=False)          # semaphore


class Model(Base):
    __tablename__= "models"
    model_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    algorithm = Column(String)
    version = Column(Integer, default=1)        # model version from MLflow
    train = Column(Integer, ForeignKey("trains.train_id"))
    service = Column(Integer, ForeignKey("model_services.model_service_id"))
    # encoder = Column(Integer, ForeignKey("train_encoders.train_encoder_id"))
    # scaler = Column(Integer, ForeignKey("train_scaler.model_scaler_id"))
    # encoder, scaler 모두 train 통해서 접근 가능
    description = Column(String, default="")


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    model_metric_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model = Column(Integer, ForeignKey("models.model_id"))
    feature_list = Column(JSON, nullable=True)
    metrics = Column(JSON)          # MAE, MAPE, accuracy, etc...
    feature_importance = Column(JSON, nullable=True)
    bias = Column(Float, nullable=True)
    default_values = Column(JSON, nullable=True)
    params = Column(JSON, nullable=True)
