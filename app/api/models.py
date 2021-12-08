from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Table, Enum, Date, JSON, DateTime, Float, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy_utils import EmailType, UUIDType

# SQLAlchemy model 생성 전에 반드시 import
from .database import Base

# 2단계 association
membership = Table('membership', Base.metadata,
                          Column('user_id', Integer, ForeignKey("users.user_id"), primary_key=True),
                          Column('product_id', Integer, ForeignKey("products.product_id"), primary_key=True),
                          Column('role', Enum("Manager","Member"))
                          )
# 1단계 association
# class Membership(Base):
#     __tablename__ = "membership"
#     user_id = Column(Integer, ForeignKey("users.user_id"), primary_key=True)
#     project_id = Column(Integer, ForeignKey("projets.project_id", primary_key=True))
#     role = Column(Enum("Manager", "Member"))
#     user = relationship("User", back_populates="membership")
#     project = relationship("Project", back_populates="membership")


class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    # role = Column()
    organization = Column(Integer, ForeignKey("organizations.organization_id"))
    username = Column(String, unique=True)
    email = Column(EmailType, unique=True)      # admin@organization.~~~
    phonenumber = Column(String, default='')    # format
    realname = Column(String, default='')       # UTF-8
    is_initiated = Column(Boolean, default=False)    # invitation pending or not
    department = Column(String, default='')
    position = Column(String, default='')
    init_date = Column(Date)                    # initiated date
    # 1단계 association
    # projects = relationship("Membership", back_populates='users')
    products = relationship("Product", secondary=membership, back_populates='users')


# Product Metadata
class ProductType(Base):
    __tablename__ = "product_types"
    product_type_id = Column(Integer, primary_key=True, index=True)
    category = Column(Enum("HR", "Univ"))  # HR / Univ (Scope) - tag in Optimizer Experiment
    product_type_name = Column(String)   # 상품 이름 (퇴사예측)
    best_model_criteria = Column(String)  # auc, f1-score ......and so on
    coverage = Column(Enum("Full", "Half", "Demo"), default="Full")
    type = Column(Enum("classification", "regression"))
    # feature_template = Column(Integer, ForeignKey("feature_templates.feature_template_id"))
    icon = Column(String)                   # product type icon (from public s3)


# feature depends on ProjectType (Template)
class FeatureTemplate(Base):
    __tablename__ = "feature_templates"
    feature_template_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    essentials = Column(JSON)           # 필수 Feature
    options = Column(JSON)              # 선택 Feature
    # feature_template = Column(JSON)  # 프로젝트 타입별로 미리 작성된 선택피쳐들 템플릿.
    # feature_added = Column(JSON, nullable=True)  # added features


# Feature depends on Project
class Feature(Base):
    __tablename__ = "features"
    feature_response_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    product = Column(Integer, ForeignKey("products.product_id"))
    feature_template = Column(Integer, ForeignKey("feature_templates.feature_template_id"))
    feature_added = Column(JSON)        # 추가 Feature


class Product(Base):
    __tablename__ = "products"
    product_id = Column(Integer, primary_key=True, index=True)
    product_type = Column(Integer, ForeignKey("product_types.product_type_id"))
    organization = Column(Integer, ForeignKey("organizations.organization_id"))
    license = Column(Integer, ForeignKey("licenses.license_id"))
    # 1 project에 n개 order가 생길 수 있음.
    last_main_order = Column(Integer, ForeignKey("main_orders.main_order_id"))

    # 1단계 association
    # users = relationship("Membership", back_populates='projects')
    users = relationship("User", secondary=membership, back_populates='products')


class License(Base):
    __tablename__ = 'licences'
    license_id = Column(Integer, primary_key=True, index=True)
    code = Column(UUIDType(binary=False), unique=True)
    predict_period = Column(Enum("monthly", "yearly", "half", "quarter"))   # 예측 주기
    start_date = Column(Date)  # license 시작 날짜
    end_date = Column(Date)    # license 끝 날짜
    organization = Column(Integer, ForeignKey("organizations.organization_id"))
    product_type = Column(Integer, ForeignKey("product_types.product_type_id"))
    description = Column(String)
    is_demo = Column(Boolean, default=False)        # default false
    is_activated = Column(Boolean, default=False)   # default false
    is_last = Column(Boolean, default=False)        # 임시. ADMIN page 에서 last license 만 보여주기 위함


# Public S3, S3 bucket은 고정. 고정된 S3 bucket에 지정된 path만 적용되면 됨.
# s3://bucket_name/organization_id/filename.png
class Organization(Base):
    __tablename__ = "organizations"
    organization_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    domain = Column(String)         # 구 url
    bi = Column("FILEFIELD", default="ALAB favicon")            # public s3 path, url
    admin_user = Column(Integer, ForeignKey("users.user_id"), nullable=True)


class Form(Base):
    __tablename__ = "forms"
    form_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    product_type = Column(Integer, ForeignKey("product_types.product_type_id"))

    # 1 form, n steps 역참조하기 위함
    steps = relationship("step_templates")


class StepTemplate(Base):
    # 이거 템플릿임.
    __tablename__ = "step_templates"
    step_template_id = Column(Integer, primary_key=True, index=True)
    sequence = Column(Integer)
    title = Column(String)
    icon = Column(Integer)      # public s3 path?
    content_title = Column(String)
    content_desc1 = Column(String, nullable=True)
    content_desc2 = Column(String, nullable=True)
    form = Column(Integer, ForeignKey("forms.form_id"))


# FormResponse 는 항상 최상단에는 last 만 보이게끔
class FormResponse(Base):
    __tablename__ = "form_responses"
    form_response_id = Column(Integer, primary_key=True, index=True)
    # feature added 가져오기위해서. license 는 바뀌어도 product 는 유지되어야함
    product = Column(Integer, ForeignKey("products.product_id"))
    feature_selected = Column(JSON, nullable=True)  # List of selected template_feature (or ARRAY)
    response = Column(JSON, nullable=True)          # 현재는 present / future 2개, 추후 변경 가능 === 분기, 월별 등등.
    status = Column(JSON)                           # Step Status List
    # Order 가 FormResponse 를 ForeignKey 로 가짐
    # FormResponse : Order = 1 : N 가능


# FormResponse 를 기반으로 Order 작성 (산출)
# Modeling 과 Dashboard 제작에 필요한 부분
class Order(Base):
    __tablename__= "orders"
    order_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # form response 는 동일해도, 여러번 학습을 하면 여러번의 Order 발생
    form_response = Column(Integer, ForeignKey("form_responses.form_response_id"))
    encrypted_feature = Column(JSON, nullable=True)
    data = Column(JSON)                 # Training Data ID, Validation Data ID, Inference Data ID
    started_at = Column(Date)           # Dashboard 는 Order(Train) 기준으로 History
    finished_at = Column(Date)          # Dashboard finished date for History
    train = Column(Integer, ForeignKey("train_id"))     # AFK. Order : Train = 1 : 1
    model = Column(Integer, ForeignKey("model_id"))     # AFK.
    product_type = Column(Integer, ForeignKey("product_types.product_type_id"))     # for OrderMediation (?)
    # Data 의 JSON Format 에 따라서 다른 종류의 train 을 호출할 수 있다면? or product_type 에 따라서?
    # 암호화 적용됨 - encrypted_feature 을 null 인지 아닌지 판별해서 적용
    # Data metadata, Model Metric 은 각각 Data App, Optimizer 에서 받아옴

# class ComplexOrder(Base):
#     __tablename__ = "main_orders"
#     main_order_id = Column(Integer, primary_key=True, index=True)
#     project = Column(Integer, ForeignKey("projects.project_id"))
#     step_response


class HROrder(Base):
    """
        T + I 에 해당하는 오더.
        필요한 정보를 모두 이미 들고 있으면 됨.
    """
    __tablename__ = "hr_orders"
    hr_order_id = Column(Integer, primary_key=True, index=True)
    project = Column(Integer, ForeignKey("projects.project_id"))
    form_response = Column(Integer, ForeignKey("form_responses.form_response_id"))
    label = Column(String)          # project_type이 가지고 있어야하는건가?
    # organization = Column(ForeignKey("organizations.organization_id"))


class HRMediator(Base):
    __tablename__ = "hr_mediators"
    hr_mediator_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    hr_order = Column(Integer, ForeignKey("hr_orders.hr_order_id"))
    train = Column(Integer, ForeignKey("train.train_id"), nullable=True)
    model = Column(Integer, ForeignKey("model.model_id"), nullable=True)
    modeling_dataset = Column(Integer, ForeignKey("uploaded_datasets.uploaded_dataset_id"), nullable=True)
    train_and_validation_dataset = Column(Integer, ForeignKey("uploaded_datasets.uploaded_dataset_id"), nullable=True)
    inference_dataset = Column(Integer, ForeignKey("uploaded_datasets.uploaded_dataset_id"), nullable=True)






    # T, S, I Order는 relationship으로 처리




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
    dataset_type = Column(String)       # training set, validation set, inference set ...
    created_at = Column(DateTime)       # Data Version
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