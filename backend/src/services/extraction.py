from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.schemas import Chunk, DocumentMetadata
from models.extraction_schema import ShipmentData, ExtractionResponse
from core.config import settings
import logging
import json

logger = logging.getLogger(__name__)

class ExtractionService:
    def __init__(self):
        try:
            self.llm = ChatGroq(
                temperature=0,
                model_name=settings.QA_MODEL,
                api_key=settings.GROQ_API_KEY
            )
            self.parser = PydanticOutputParser(pydantic_object=ShipmentData)
            
            self.extraction_prompt = ChatPromptTemplate.from_template(
                """You are an expert data extraction assistant for logistics documents.
Extract ALL available information from the text provided below. Be thorough and extract:
- Reference IDs, Load IDs, PO numbers
- Shipper, Consignee, Carrier details (name, MC number, phone, email)
- Driver information (name, phone, truck/trailer numbers)
- Pickup and Drop locations (name, address, city, state, zip, appointment times)
- Dates (shipping, delivery, created, booking)
- Equipment details (type, size, load type)
- Commodities (name, weight, quantity, description)
- Rate information (total, currency, breakdown)
- Instructions (special, shipper, carrier)
- Dispatcher information (name, phone, email)

{format_instructions}

Text:
{text}

If a field is not present in the document, return null for that field.
Extract as much detail as possible from the document.
"""
            )
            logger.info("ExtractionService initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ExtractionService: {str(e)}", exc_info=True)
            raise

    def extract_data(self, text: str, filename: str = "unknown") -> ExtractionResponse:
        """
        Extract structured data from the provided text.
        Returns ExtractionResponse with extracted data.
        """
        chain = self.extraction_prompt | self.llm | self.parser
        
        try:
            logger.debug(f"Attempting extraction from text of length {len(text)}")
            result = chain.invoke({
                "text": text,
                "format_instructions": self.parser.get_format_instructions()
            })
            logger.info("Successfully extracted data.")
            return ExtractionResponse(data=result, document_id=filename)
        except Exception as e:
            logger.error(f"Extraction error: {e}", exc_info=True)
            # Return empty structure on error
            return ExtractionResponse(data=ShipmentData(), document_id=filename)

    def format_extraction_as_text(self, extraction: ExtractionResponse) -> str:
        """
        Format extracted data as human-readable text for vector storage.
        This makes the structured data searchable.
        """
        data = extraction.data
        lines = ["=== EXTRACTED STRUCTURED DATA ===\n"]
        
        # Core IDs
        if data.reference_id:
            lines.append(f"Reference ID: {data.reference_id}")
        if data.load_id:
            lines.append(f"Load ID: {data.load_id}")
        if data.po_number:
            lines.append(f"PO Number: {data.po_number}")
        
        # Parties
        if data.shipper:
            lines.append(f"\nShipper: {data.shipper}")
        if data.consignee:
            lines.append(f"Consignee: {data.consignee}")
        
        # Carrier
        if data.carrier:
            lines.append(f"\nCarrier Name: {data.carrier.carrier_name}")
            if data.carrier.mc_number:
                lines.append(f"MC Number: {data.carrier.mc_number}")
            if data.carrier.phone:
                lines.append(f"Carrier Phone: {data.carrier.phone}")
        
        # Driver
        if data.driver:
            lines.append(f"\nDriver Name: {data.driver.driver_name}")
            if data.driver.cell_number:
                lines.append(f"Driver Phone: {data.driver.cell_number}")
            if data.driver.truck_number:
                lines.append(f"Truck Number: {data.driver.truck_number}")
        
        # Locations
        if data.pickup:
            lines.append(f"\nPickup Location: {data.pickup.name or data.pickup.address or 'N/A'}")
            if data.pickup.city:
                lines.append(f"Pickup City: {data.pickup.city}, {data.pickup.state or ''}")
            if data.pickup.appointment_time:
                lines.append(f"Pickup Appointment: {data.pickup.appointment_time}")
        
        if data.drop:
            lines.append(f"\nDrop Location: {data.drop.name or data.drop.address or 'N/A'}")
            if data.drop.city:
                lines.append(f"Drop City: {data.drop.city}, {data.drop.state or ''}")
        
        # Dates
        if data.shipping_date:
            lines.append(f"\nShipping Date: {data.shipping_date}")
        if data.delivery_date:
            lines.append(f"Delivery Date: {data.delivery_date}")
        
        # Equipment
        if data.equipment_type:
            lines.append(f"\nEquipment Type: {data.equipment_type}")
        if data.equipment_size:
            lines.append(f"Equipment Size: {data.equipment_size} feet")
        if data.load_type:
            lines.append(f"Load Type: {data.load_type}")
        
        # Commodities
        if data.commodities:
            lines.append("\nCommodities:")
            for i, commodity in enumerate(data.commodities, 1):
                lines.append(f"  {i}. {commodity.commodity_name or 'Unknown'}")
                if commodity.weight:
                    lines.append(f"     Weight: {commodity.weight}")
                if commodity.quantity:
                    lines.append(f"     Quantity: {commodity.quantity}")
        
        # Rate
        if data.rate_info:
            lines.append(f"\nTotal Rate: ${data.rate_info.total_rate} {data.rate_info.currency or 'USD'}")
            if data.rate_info.rate_breakdown:
                lines.append(f"Rate Breakdown: {json.dumps(data.rate_info.rate_breakdown)}")
        
        # Instructions
        if data.special_instructions:
            lines.append(f"\nSpecial Instructions: {data.special_instructions}")
        if data.shipper_instructions:
            lines.append(f"Shipper Instructions: {data.shipper_instructions}")
        if data.carrier_instructions:
            lines.append(f"Carrier Instructions: {data.carrier_instructions}")
        
        # Dispatcher
        if data.dispatcher_name:
            lines.append(f"\nDispatcher: {data.dispatcher_name}")
            if data.dispatcher_phone:
                lines.append(f"Dispatcher Phone: {data.dispatcher_phone}")
        
        return "\n".join(lines)

    def create_structured_chunk(self, extraction: ExtractionResponse, filename: str) -> Chunk:
        """
        Create a Chunk object from extracted data for vector storage.
        """
        formatted_text = self.format_extraction_as_text(extraction)
        
        metadata = DocumentMetadata(
            filename=filename,
            chunk_id=9999,  # Special ID for structured data chunks
            source=f"{filename} - Extracted Data",
            chunk_type="structured_data"
        )
        
        return Chunk(text=formatted_text, metadata=metadata)

extraction_service = ExtractionService()
